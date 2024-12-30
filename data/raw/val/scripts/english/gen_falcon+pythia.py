#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("export HF_HOME='./.hf'")
get_ipython().system("export TRANSFORMERS_CACHE='./.hf'")
get_ipython().system("export TRANSFORMERS_HOME='./.hf'")
get_ipython().system("export HF_CACHE='./.hf'")

import os
os.environ['HF_HOME'] = './.hf/'


import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

file_path = "english-with-questions-valid+test.tsv"
records = pd.read_csv(file_path, sep='\t').to_dict(orient='records')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


pd.read_csv(file_path, sep='\t')


# In[3]:


# model_name = 'togethercomputer/Pythia-Chat-Base-7B' 
model_name = 'tiiuae/falcon-7b-instruct' 

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

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)



# In[4]:


import tqdm.notebook as tqdm
from transformers.utils import logging
import pathlib
logging.set_verbosity_warning()

for shorthand, config in tqdm.tqdm(configs):
    output_file_path = f'english-{model_name.split("/")[1]}.{shorthand}.jsonl'
    anootation_file_path = f'english-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl'
    if not pathlib.Path(anootation_file_path).is_file():
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for record in tqdm.tqdm(records):
                message = record['question']
                # prompt = f"<human>: {message}\n<bot>:"
                prompt = message + '\n'
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    # eos_token_id=tokenizer.encode('\n'),
                    # pad_token_id=tokenizer.encode('\n')[0],
                    **config,
                )
        
                response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response_text = response_text.replace(prompt, "") # response repeats the input in the begining
                response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l.to("cpu").tolist() for l in outputs.logits]
                
                """print("\n\n")
                print(f"Q: {message}")
                print(f"A: {response_text}")
        
                print("input length", len(inputs.input_ids[0]))
                # print("sequence length", len(outputs.sequences[0]))
                print("response token length", len(response_tokens))
                print("response token ID length", len(response_token_ids))
                print("logits length", len(response_logits))
                # print("embedding length", len(response_embeddings))
                raise"""
        
        
                record['model_id'] = model_name
                record['lang'] = 'EN'
        
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits
                # record['output_embeddings'] = response_embeddings
        
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        
        columns_to_extract = ['en-url', 'lang', 'question', 'model_id', 'output_text', 'output_tokens']
        
        output_data = []
        
        with open(anootation_file_path, 'w', encoding='utf-8') as file:
            for data in records:
                extracted_data = {key: data[key] for key in columns_to_extract if key in data}
        
                json.dump(extracted_data, file, ensure_ascii=False)
                file.write('\n')


# In[5]:


del model

model_name = 'togethercomputer/Pythia-Chat-Base-7B' 

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

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)



# In[6]:


import tqdm.notebook as tqdm
from transformers.utils import logging
import pathlib
logging.set_verbosity_warning()

for shorthand, config in tqdm.tqdm(configs):
    output_file_path = f'english-{model_name.split("/")[1]}.{shorthand}.jsonl'
    anootation_file_path = f'english-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl'
    if not pathlib.Path(anootation_file_path).is_file():
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for record in tqdm.tqdm(records):
                message = record['question']
                prompt = f"<human>: {message}\n<bot>:"
                # prompt = message + '\n'
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    # eos_token_id=tokenizer.eos_token_id,
                    # pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.encode('\n'),
                    pad_token_id=tokenizer.encode('\n')[0],
                    **config,
                )
        
                response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response_text = response_text.replace(prompt, "") # response repeats the input in the begining
                response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l.to("cpu").tolist() for l in outputs.logits]
                
                """print("\n\n")
                print(f"Q: {message}")
                print(f"A: {response_text}")
        
                print("input length", len(inputs.input_ids[0]))
                # print("sequence length", len(outputs.sequences[0]))
                print("response token length", len(response_tokens))
                print("response token ID length", len(response_token_ids))
                print("logits length", len(response_logits))
                # print("embedding length", len(response_embeddings))
                raise"""
        
        
                record['model_id'] = model_name
                record['lang'] = 'EN'
        
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits
                # record['output_embeddings'] = response_embeddings
        
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        
        columns_to_extract = ['en-url', 'lang', 'question', 'model_id', 'output_text', 'output_tokens']
        
        output_data = []
        
        with open(anootation_file_path, 'w', encoding='utf-8') as file:
            for data in records:
                extracted_data = {key: data[key] for key in columns_to_extract if key in data}
        
                json.dump(extracted_data, file, ensure_ascii=False)
                file.write('\n')

