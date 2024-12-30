#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

file_path = "questions-de.tsv"
records = pd.read_csv(file_path, sep='\t').to_dict(orient='records')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


pd.read_csv(file_path, sep='\t')


# In[4]:


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


# In[5]:


# model_name = 'togethercomputer/Pythia-Chat-Base-7B' 
model_name = 'malteos/bloom-6b4-clp-german-oasst-v0.1' 

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



# In[ ]:


import tqdm.notebook as tqdm
from transformers.utils import logging
import pathlib
logging.set_verbosity_warning()

for shorthand, config in tqdm.tqdm(configs):
    output_file_path = f'german-{model_name.split("/")[1]}.{shorthand}.jsonl'
    anootation_file_path = f'german-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl'
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
                record['lang'] = 'DE'
        
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits
                # record['output_embeddings'] = response_embeddings
        
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        
        columns_to_extract = ['URL-de', 'lang', 'question', 'model_id', 'output_text', 'output_tokens', 'title']
        
        output_data = []
        
        with open(anootation_file_path, 'w', encoding='utf-8') as file:
            for data in records:
                extracted_data = {key: data[key] for key in columns_to_extract if key in data}
        
                json.dump(extracted_data, file, ensure_ascii=False)
                file.write('\n')


# In[5]:


del model


# In[5]:


from transformers import AutoTokenizer, MistralForCausalLM, set_seed
model_name = "occiglot/occiglot-7b-de-en-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MistralForCausalLM.from_pretrained(model_name).to(device)


# In[9]:


import tqdm.notebook as tqdm
from transformers.utils import logging
import pathlib
logging.set_verbosity_warning()
#model = model.to(device)

for shorthand, config in tqdm.tqdm(configs):
    output_file_path = f'german-{model_name.split("/")[1]}.{shorthand}.jsonl'
    anootation_file_path = f'german-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl'
    if not pathlib.Path(anootation_file_path).is_file():
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for record in tqdm.tqdm(records):
                messages = [
                   {"role": "system", 'content': 'You are a helpful assistant. Please give short and concise answers.'},
                   {"role": "user", "content": record['question']},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_dict=False, 
                    return_tensors='pt',
                ).to(device)
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
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
        
                response_text = tokenizer.decode(outputs.sequences[0][len(inputs[0]):], skip_special_tokens=True)
                # response_text = response_text.replace(prompt, "") # response repeats the input in the begining
                response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs[0]):]
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
                record['lang'] = 'DE'
        
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits
                # record['output_embeddings'] = response_embeddings
        
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        
        columns_to_extract = ['URL-de', 'lang', 'question', 'model_id', 'output_text', 'output_tokens', 'title']
        
        output_data = []
        
        with open(anootation_file_path, 'w', encoding='utf-8') as file:
            for data in records:
                extracted_data = {key: data[key] for key in columns_to_extract if key in data}
        
                json.dump(extracted_data, file, ensure_ascii=False)
                file.write('\n')

