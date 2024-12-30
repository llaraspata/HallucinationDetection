import json
import csv
import sys
import torch
import random
import argparse
import pandas as pd
import tqdm.notebook as tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ind", type=int, help="Stating index for data selection")
    parser.add_argument("--end_ind", type=int, help="End index for data selection")
    parser.add_argument("--n_out_file", type=int, help="Outfile number")

    args = parser.parse_args()
    return args


def read_data(file_path):
    with open(file_path, 'r') as istr:
        reader = csv.reader(istr, delimiter='\t')
        header = next(reader)
        records = [dict(zip(header, row)) for row in reader]
    return records


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir="./tmp",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./tmp")
    print("Done loading!")
    return model, tokenizer


def write_records(record, file, logprobs_file, include_logits=False):
    records = record.copy()
    if include_logits:
        json.dump(records, logprobs_file, ensure_ascii=False)
        logprobs_file.write("\n")
    else:
        json.dump(records, file, ensure_ascii=False)
        file.write("\n")


def main():
    args = parse_args()
    # records = read_data('multiparallel-fi-with-questions-filtered.tsv')
    records = read_data("multiparallel-fi-with-questions-missing-2.tsv")
    start = args.start_ind
    end = args.end_ind
    records = records[start:end]

    model_name = "LumiOpen/Poro-34B-chat"
    model, tokenizer = load_model(model_name)

    save_records = []

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

    with open(f'outputs/poro-answers-with-logprobs-{args.n_out_file}.jsonl', 'w') as ostr_logprobs, \
    open(f'outputs/poro-answers-{args.n_out_file}.jsonl', 'w') as file:

        with torch.no_grad():
            for record in tqdm.tqdm(records, desc='items'):

                seed = random.randint(1, 10000)
                set_seed(seed)

                config_name, config = random.choice(configs)

                message = record['question'].rstrip('\n')
                message = [{"role": "user", "content": message}]
                prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                output = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    )
                response_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                response_text = response_text.replace(prompt, "") # response repeats the input in the begining

                new_text_marker = "assistant\n"
                marker_index = response_text.find(new_text_marker)
                if marker_index != -1:
                    response_text = response_text[marker_index + len(new_text_marker):].strip()

                response_token_ids = output.sequences[0].to("cpu").tolist()[len(input_ids[0]):]
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                
                response_logits = []
                for i, token_id in enumerate(response_token_ids):
                    token_logits = output.scores[i].to("cpu")[0, token_id].item()
                    response_logits.append(token_logits)

                record.update({
                    "model_id": model_name,
                    "lang": "FI",
                    "output_text": response_text,
                    "output_tokens": response_tokens,
                    "config": config_name,
                })
                write_records(record, file, ostr_logprobs)

                record['output_logits'] = response_logits
                write_records(record, file, ostr_logprobs, include_logits=True)

                columns_to_extract = ["URL-fi", "lang", "question", "model_id", "output_text", "output_tokens", "title"]
                extracted_data = {key: record[key] for key in columns_to_extract if key in record}
                save_records.append(extracted_data)

    with open(f"outputs/annotation-poro-{args.n_out_file}.jsonl", "w", encoding="utf-8") as f:
        for d in save_records:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    main()
