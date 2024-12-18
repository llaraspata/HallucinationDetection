import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "Qwen/Qwen1.5-14B-Chat"

file_path = "chinese.jsonl"
with open(file_path, 'r') as file:
    records = [json.loads(line) for line in file]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

output_file_path = f'chinese-{model_name.split("/")[1]}.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for record in records:
        message = record['input_question']
        if 'internlm' in model_name:
            prompt = f"[INST] {message} [/INST]"
        else:
            prompt = f"{message}"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.6,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=True
        )

        response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response_text = response_text.replace(prompt, "") # response repeats the input in the begining
        response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
        response_embeddings = [x.to("cpu").tolist() for x in outputs.hidden_states[0][-1][0]] # embedding of the last layer
        response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
        response_logits = [l.to("cpu").tolist() for l in outputs.logits]
        
        print("\n\n")
        print(f"Q: {message}")
        print(f"A: {response_text}")

        print("input length", len(inputs.input_ids[0]))
        print("embedding length", len(response_embeddings))
        print("response token length", len(response_tokens))
        print("response token ID length", len(response_token_ids))
        print("logits length", len(response_logits))

        record['model_id'] = model_name

        record['output_text'] = response_text
        record['output_tokens'] = response_tokens
        record['output_logits'] = response_logits
        record['output_embeddings'] = response_embeddings

        json.dump(record, file, ensure_ascii=False)
        file.write('\n')

anootation_file_path = f'chinese-{model_name.split("/")[1]}-annotation.jsonl'

columns_to_extract = ['url', 'lang', 'input_question', 'translated_question', 'model_id', 'output_text', 'output_tokens']

output_data = []

with open(anootation_file_path, 'w', encoding='utf-8') as file:
    for data in records:
        extracted_data = {key: data[key] for key in columns_to_extract if key in data}

        json.dump(extracted_data, file, ensure_ascii=False)
        file.write('\n')
