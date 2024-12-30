import json
import csv
import torch
import pandas as pd
import tqdm

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
###########################
# THIS MODEL IS NOT REALLY RELIABLE IN ITS OUTPUT 
# I DID NOT FIND ANY WORKING EXAMPLE ON "HOT TO USE IT" ONLINE
###########################
set_seed(94326)

def read_data(file_path):
    with open(file_path, 'r') as istr:
        reader = csv.reader(istr)
        header = next(reader)
        records = [dict(zip(header, row)) for row in reader]

    return records

def load_model(model_name):
    access_token = "YOUR_ACCESS_TOKEN_HERE"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", token = access_token, load_in_8bit=True #torch_dtype=torch.bfloat16
    )
    tokname = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(tokname,  token = access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def main():
    pathdata = "./it-mushroom.val2.csv"
    records = read_data(pathdata)

    model_name = "rstless-research/DanteLLM-7B-Instruct-Italian-v0.1"

    model, tokenizer = load_model(model_name)
    print(tokenizer.eos_token_id)
    configs = [
        ('k50_t0.1', dict(top_k=50, temperature=0.1)),
        ('k50_t0.2', dict(top_k=50, temperature=0.2)),
        ('k50_t0.5', dict(top_k=50, temperature=0.5)),
        ('k50_t1.0', dict(top_k=50, temperature=1.0)),
        ('k75_t0.1', dict(top_k=75, temperature=0.1)),
        ('k75_t0.2', dict(top_k=75, temperature=0.2)),
        ('k75_t0.5', dict(top_k=75, temperature=0.5)),
        ('k75_t1.0', dict(top_k=75, temperature=1.0)),
        ('k100_t0.1', dict(top_k=100, temperature=0.1)),
        ('k100_t0.2', dict(top_k=100, temperature=0.2)),
        ('k100_t0.5', dict(top_k=100, temperature=0.5)),
        ('k100_t1.0', dict(top_k=100, temperature=1.0)),
    ]

    for shorthand, config in tqdm.tqdm(configs, desc='configs'):
        with open(f'./dante-answers.{shorthand}.jsonl', 'w') as file:
            for record_ in tqdm.tqdm(records, desc='items'):
                record = {k: v for k, v in record_.items()}
                message = record['IT questions'].strip()
                message = [{"role": "user", "content": "Ciao chi sei?"}, {"role": "assistant", "content": "Ciao, sono DanteLLM, un large language model. Come posso aiutarti?"}, {"role": "user", "content": message}]
                prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                #terminators = [
                #    tokenizer.eos_token_id,
                #    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                #]
                output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    **config,
                )
                # print("output:", output)
                # response_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
                # response_text = response_text.replace(prompt, "", 1) # response repeats the input in the begining
                response_token_ids = output.sequences[0].to("cpu").tolist()[len(input_ids[0]):]
                # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)
                response_logits = [l.to("cpu").tolist() for l in output.logits]
                #print("prompt:", prompt)
                #print("response_text:", response_text)
                #print("response_logits:", len(response_logits))
                #print("response_tokens:", len(response_tokens))
                record['model_id'] = model_name
                record['lang'] = 'IT'
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits

                json.dump(record, file, ensure_ascii=False)
                file.write('\n')

if __name__ == "__main__":
    main()
