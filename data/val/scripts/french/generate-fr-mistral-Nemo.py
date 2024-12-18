import json
import csv
import torch
import pandas as pd
import tqdm

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

set_seed(94326)


def read_data(file_path):
    with open(file_path, 'r') as istr:
        reader = csv.reader(istr)
        header = next(reader)
        records = [dict(zip(header, row)) for row in reader]

    return records

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=".hf",
        token='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',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token='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',
    )
    return model, tokenizer

def main(args):
    records = read_data('fr-val-questions-batch2.csv')

    model_name = "mistralai/Mistral-Nemo-Instruct-2407"

    model, tokenizer = load_model(model_name)

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

    for config_idx in tqdm.tqdm(args.configs, desc='configs'):
        shorthand, config = configs[config_idx]
        with open(f'outputs/mistralNemo-answers.{shorthand}.batch2.jsonl', 'w') as file:
            for record_ in tqdm.tqdm(records, desc=shorthand):
                record = {k: v for k, v in record_.items()}
                message = record['FR questions'].strip()
                message = [{"role": "user", "content": message}]
                prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
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
                record['lang'] = 'FR'
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits

                json.dump(record, file, ensure_ascii=False)
                file.write('\n')

if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument('configs', nargs='+', type=int)
    args = p.parse_args()
    main(args)
