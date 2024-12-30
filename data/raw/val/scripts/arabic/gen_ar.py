import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm.notebook as tqdm
from transformers.utils import logging
import json

logging.set_verbosity_warning()

seed = 42
torch.manual_seed(seed)


records = pd.read_csv("questions-ar.csv")
records= records.to_dict(orient='records')


# Display the contents of the second sheet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_names = [
    "SeaLLMs/SeaLLM-7B-v2.5",
    "openchat/openchat-3.5-0106-gemma",
    "arcee-ai/Arcee-Spark",
]

split_text_array = ["\n<|im_start|>assistant\n", "\nassistant", "\nassistant"]
configs = [
    ("k50_p0.90_t0.1", dict(top_k=50, top_p=0.90, temperature=0.1)),
    #  ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ("k50_p0.90_t0.2", dict(top_k=50, top_p=0.90, temperature=0.2)),
    # ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ("k50_p0.90_t0.3", dict(top_k=50, top_p=0.90, temperature=0.3)),
    # ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
    ("k75_p0.90_t0.1", dict(top_k=75, top_p=0.90, temperature=0.1)),
    # ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ("k75_p0.90_t0.2", dict(top_k=75, top_p=0.90, temperature=0.2)),
    # ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ("k75_p0.90_t0.3", dict(top_k=75, top_p=0.90, temperature=0.3)),
    # ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
]

model_idx = 2
model_name = model_names[model_idx]
split_text = split_text_array[model_idx]

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


print("Model used ", model_name)
model_short = model_name.split("/")[-1]

for record in tqdm.tqdm(records):
    i += 1
    print(
        f"Link in Arabic: {record['Link in Arabic']}"
    )  # Print 'Link in Arabic' once for every sample
    question = str(record["AR questions"])
    print(question)
    generated_answers = []
    configs_to_show = list(configs)[:5]  # Ensure only 5 configs are shown

    for shorthand, config in tqdm.tqdm(configs_to_show):

        messages = [
            {"role": "user", "content": "أجب عن السؤال التالي بشكل دقيق ومختصر"},
            {
                "role": "assistant",
                "content": "بالطبع! ما هو السؤال الذي تود الإجابة عنه؟",
            },
            {"role": "user", "content": question},
        ]
        encodeds = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )

        model_inputs = encodeds.to(device)

        outputs = model.generate(
            model_inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            **config,
        )

        response_text = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        ).strip()
        response_text = response_text.split(split_text)[-1].strip()
        print(response_text)
        response_token_ids = (
            outputs.sequences[0].to("cpu").tolist()[len(model_inputs[0]) :]
        )
        response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
        response_logits = [l.to("cpu").tolist() for l in outputs.logits]

        # Extract only the logits corresponding to the output tokens
        response_logits = [logit.to("cpu").tolist() for logit in outputs.logits]

        generated_answers.append((shorthand, response_text, response_logits))

    # Show all 5 answers and let the user pick one
    for idx, (shorthand, answer, logits) in enumerate(generated_answers):
        print(f"\nAnswer {idx + 1} (Config: {shorthand}):\n{answer}\n")

    choice = int(input("Choose the answer to save (1-5): ")) - 1

    selected_shorthand, selected_answer, selected_logits = generated_answers[choice]

    output_file_path = f'./SHROOM/{i}/{model_name.split("/")[1]}/arabic-{model_name.split("/")[1]}.{i}.{selected_shorthand}.jsonl'
    os.makedirs(
        f'./SHROOM/{i}/{model_name.split("/")[1]}', exist_ok=True
    )
    os.makedirs(f"/./SHROOM/{i}", exist_ok=True)

    # Save the selected answer and its logits
    with open(output_file_path, "w", encoding="utf-8") as file:
        record["model_id"] = model_name
        record["lang"] = "AR"
        record["output_text"] = selected_answer
        record["output_tokens"] = tokenizer.convert_ids_to_tokens(response_token_ids)
        record["output_logits"] = selected_logits

        columns_to_extract = [
            "Link in Arabic",
            "lang",
            "AR questions",
            "model_id",
            "output_text",
            "output_tokens",
        ]
        records_small = {k: record[k] for k in columns_to_extract}
        records_small["gen_config"] = config
        json.dump(records_small, file, ensure_ascii=False)
        file.write("\n")

