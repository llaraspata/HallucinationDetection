#!/usr/bin/env python
# coding: utf-8

# Installing dependencies. You might need to tweak the CMAKE_ARGS for the `llama-cpp-python` pip package.

# In[1]:


# GPU llama-cpp-python; Starting from version llama-cpp-python==0.1.79, it supports GGUF
# !CMAKE_ARGS="-DLLAMA_CUBLAS=on " pip install 'llama-cpp-python>=0.1.79' --force-reinstall --upgrade --no-cache-dir
# For download the models
# !pip install huggingface_hub
# !pip install datasets


# We start by downloading an instruction-finetuned Mistral model.

# In[9]:


from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/SauerkrautLM-7B-v1-GGUF"
model_basename = "sauerkrautlm-7b-v1.Q4_K_M.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# This config has been tested on an RTX 3080 (VRAM of 16GB).
# you might need to tweak with respect to your hardware.
from llama_cpp import Llama
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=4, # CPU cores
    n_batch=8192, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=400, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=8192, # Context window
    logits_all=True
)


# In[7]:


import tqdm.notebook as tqdm
import json 
import csv

with open('questions-de.tsv', 'r') as istr:
    reader = csv.reader(istr, delimiter='\t')
    header = next(reader)
    records = [dict(zip(header, row)) for row in reader]


# In[ ]:


import random

configs = [
    ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ('k50_p0.90_t0.3', dict(top_k=50, top_p=0.90, temperature=0.3)),
    ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
    ('k75_p0.90_t0.1', dict(top_k=75, top_p=0.90, temperature=0.1)),
    ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ('k75_p0.90_t0.2', dict(top_k=75, top_p=0.90, temperature=0.2)),
    ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ('k75_p0.90_t0.3', dict(top_k=75, top_p=0.90, temperature=0.3)),
    ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
]

random.shuffle(configs)

for shorthand, config in tqdm.tqdm(configs, desc='configs'):
    with open(f'mistral-answers-with-logprobs.{shorthand}.jsonl', 'w') as ostr_logprobs, \
    open(f'mistral-answers.{shorthand}.jsonl', 'w') as ostr:
        for record in tqdm.tqdm(records, desc='items'):
            message = record['question']
            prompt = f"[INST] {message} [/INST]"
            if 'alt_question' in record:
                del record['alt_question']
            response = lcpp_llm(
                prompt=prompt,
                logprobs=32_000,
                max_tokens=512,
                **config
            )
            print(
                json.dumps(
                    {
                        **record, 
                        'model_output': response['choices'][0]['text'],
                        'tokens': response['choices'][0]['logprobs']['tokens'],
                        'logprobs': [
                            {k: float(v) for k,v in d.items()} 
                            for d in response['choices'][0]['logprobs']['top_logprobs']
                        ],
                        'lang': 'DE',
                    }
                ), 
                file=ostr_logprobs,
            )
            
            print(
                json.dumps(
                    {
                        **record, 
                        'model_output': response['choices'][0]['text'],
                        'tokens': response['choices'][0]['logprobs']['tokens'],
                        'lang': 'EN',
                    }
                ), 
                file=ostr,
                flush=True,
            )

