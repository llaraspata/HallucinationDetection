import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import src.model.utils as ut
from src.model.SnyderEtAlProbing import SnyderEtAlProbing
from src.data.MushroomDataset import MushroomDataset
from src.data.HaluEvalDataset import HaluEvalDataset
from src.data.HaluBenchDataset import HaluBenchDataset
from src.model.prompts import PROMPT_CORRECT as prompt
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Competitor libraries
import re
import pickle
import functools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from string import Template
from typing import Any, Dict
from datetime import datetime
from functools import partial
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter

class SnyderEtAlEarlyDetection:
    # -------------
    # Constants
    # -------------
    # Data related params
    iteration = 0
    interval = 2500 # We run the inference on these many examples at a time to achieve parallelization
    start = iteration * interval
    end = start + interval
    trex_data_to_question_template = {
        "capitals": Template("What is the capital of $source?"),
        "place_of_birth": Template("Where was $source born?"),
        "founders": Template("Who founded $source?"),
    }

    # Hardware
    gpu = "0"
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Integrated Grads
    ig_steps = 64
    internal_batch_size = 4

    # Model
    layer_number = -1
    # hardcode below,for now. Could dig into all models but they take a while to load
    model_num_layers = {
        "falcon-40b" : 60,
        "falcon-7b" : 32,
        "open_llama_13b" : 40,
        "open_llama_7b" : 32,
        "opt-6.7b" : 32,
        "opt-30b" : 48,
    }
    coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
    model_repos = {
        "falcon-40b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
        "falcon-7b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
        "open_llama_13b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "open_llama_7b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "opt-6.7b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
        "opt-30b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj", ),
    }

    # For storing results
    fully_connected_hidden_layers = defaultdict(list)
    attention_hidden_layers = defaultdict(list)
    attention_forward_handles = {}
    fully_connected_forward_handles = {}
    
    TARGET_LAYERS = list(range(0, 32))     # Upper bound excluded
    MAX_NEW_TOKENS = 100
    DEFAULT_DATASET = "mushroom"
    CACHE_DIR_NAME = os.path.join("activation_cache", "snyder_et_al")
    ACTIVATION_TARGET = ["logits", "fully", "attn", "attribution"]
    PREDICTION_DIR = os.path.join("predictions", "snyder_et_al")
    RESULTS_DIR = os.path.join("results", "snyder_et_al")
    PREDICTIONS_FILE_NAME = "snyder_et_al_prediction.jsonl"
    PREDICTIONS_LAYER_FILE_NAME = "snyder_et_al_prediction_layer{layer}.jsonl"
    LABELS = {
        0: "not_hallucinated",
        1: "hallucinated"
    }

    # -------------
    # Constructor
    # -------------
    def __init__(self, project_dir):
        self.project_dir = project_dir

    
    def load_dataset(self, dataset_name=DEFAULT_DATASET, use_local=False, label=1):
        print("--"*50)
        print(f"Loading dataset {dataset_name}")
        print("--"*50)
        self.label = label
        
        if dataset_name == "mushroom":
            self.dataset_name = dataset_name
            val_path = os.path.join(self.project_dir, "data", "processed", "labeled.jsonl")
            self.dataset = MushroomDataset(data_path=val_path)

        elif dataset_name == "halu_eval":
            self.dataset_name = dataset_name
            self.dataset = HaluEvalDataset(use_local=use_local, label=label)

        elif dataset_name == "halu_bench":
            self.dataset_name = dataset_name
            self.dataset = HaluBenchDataset(use_local=use_local, label=label)

        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")


    def load_llm(self, llm_name, use_local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        print("--"*50)
        print(f"Loading LLM {llm_name}")
        print("--"*50)
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.llm = ut.load_llm(llm_name, ut.create_bnb_config(), local=use_local, dtype=dtype, use_device_map=use_device_map, use_flash_attention=use_flash_attn)
        print("--"*50)

    
    def load_hal_probing(self, activation="logits", layer=0):
        print("--"*50)
        print(f"Loading Hallucination probing model (Snyder et. al)")
        print("--"*50)
        self.probing_model = SnyderEtAlProbing(self.project_dir, activation=activation, layer=layer)
        print("--"*50)


    # -------------
    # Main Methods
    # -------------
    @torch.no_grad()
    def predict_llm(self, llm_name, data_name=DEFAULT_DATASET, label=1, use_local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)
        self.load_llm(llm_name, use_local=use_local, dtype=dtype, use_device_map=use_device_map, use_flash_attn=use_flash_attn)

        print("--"*50)
        print("Hallucination Detection - Saving LLM's activations")
        print("--"*50)
        
        print("\n0. Prepare folders")
        self._create_folders_if_not_exists(label=label)
    
        print(f"\n1. Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.compute_and_save_results()
        
        print("--"*50)

    
    @torch.no_grad()
    def predict_kc(self, llm_name, activation="logits", layer=0, data_name=DEFAULT_DATASET, use_local=False, label=1):
        self.load_hal_probing(activation=activation, layer=layer)
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)

        print("--"*50)
        print("Hallucination Detection - Saving Hallucination Probing Predictions")
        print(f"Label: {self.LABELS[label]}")
        print("--"*50)
        
        with open(self.results_pkl_path, "rb") as infile:
            activations = pickle.loads(infile.read())

        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name)
        
        n_instances = len(activations['instance_id'])

        preds = []

        if activation == "logits" or activation == "attribution":
            preds_filename = self.PREDICTIONS_FILE_NAME

            for i in tqdm(range(n_instances), desc="Predicting"):
                instance_id = activations['instance_id'][i]

                pred = {
                    "instance_id": instance_id,
                    "lang": self.dataset.get_language_by_instance_id(instance_id),
                    "prediction": self.probing_model.predict(activations[activation][i]).item(),
                    "label": label
                }

                preds.append(pred)
        else:
            preds_filename = self.PREDICTIONS_LAYER_FILE_NAME
            
            for i in tqdm(range(n_instances), desc="Predicting"):
                instance_id = activations['instance_id'][i][layer]

                pred = {
                    "instance_id": instance_id,
                    "lang": self.dataset.get_language_by_instance_id(instance_id),
                    "prediction": self.probing_model.predict(activations[activation][i][layer]).item(),
                    "label": label
                }

                preds.append(pred)

        path_to_save = os.path.join(result_path, self.dataset_name, activation, preds_filename)

        if not os.path.exists(os.path.dirname(path_to_save)):
            os.makedirs(os.path.dirname(path_to_save))
        else:
            # If the file already exists, load existing predictions and add the new ones
            if os.path.exists(path_to_save):
                existing_preds = json.load(open(path_to_save, "r"))
                preds.extend(existing_preds)

        json.dump(preds, open(path_to_save, "w"), indent=4)
        
        print(f"\t -> Predictions saved to {path_to_save}")


    def eval(self, target, llm_name, data_name=DEFAULT_DATASET):
        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name)
        print("--"*50)
        print("Hallucination Detection - Evaluation")
        print(f"Activation: {target}, Layer: {self.kc_layer}")
        print("--"*50)

        print(f"\n1. Load predictions for layer {self.kc_layer}")
        preds_path = os.path.join(result_path, data_name, target, self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))
        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"Predictions file not found: {preds_path}")
        
        preds = json.load(open(preds_path, "r"))

        print("\n2. Compute metrics")
        metrics = SnyderEtAlEarlyDetection.compute_all_metrics(preds, data_name)

        print("\n3. Save results")
        self._save_metrics(metrics, target, data_name, llm_name)        
        
        print("--"*50)


    # -------------
    # Public Methods
    # -------------
    def save_fully_connected_hidden(layer_name, mod, inp, out):
        SnyderEtAlEarlyDetection.fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


    def save_attention_hidden(layer_name, mod, inp, out):
        SnyderEtAlEarlyDetection.attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


    def get_stop_token(self):
        if "llama" in self.llm_name:
            stop_token = 13
        elif "falcon" in self.llm_name:
            stop_token = 193
        else:
            stop_token = 50118
        return stop_token


    def get_next_token(x, model):
        with torch.no_grad():
            return model(x).logits


    def generate_response(self, x, model, *, max_length=100, pbar=False):
        response = []
        bar = tqdm(range(max_length)) if pbar else range(max_length)
        for step in bar:
            logits = SnyderEtAlEarlyDetection.get_next_token(x, model)
            next_token = logits.squeeze()[-1].argmax()
            x = torch.concat([x, next_token.view(1, -1)], dim=1)
            response.append(next_token)
            if next_token == self.get_stop_token() and step>5:
                break
        return torch.stack(response).cpu().numpy(), logits.squeeze()


    def answer_question(self, question, answer, model, tokenizer, *, max_length=100, pbar=False):
        model_input = prompt.format(question=question, answer=answer)
        input_ids = tokenizer(model_input, return_tensors='pt').input_ids.to(model.device)
        response, logits = self.generate_response(input_ids, model, max_length=max_length, pbar=pbar)
        return response, logits, input_ids.shape[-1]


    def answer_trivia(self, question, answer, model, tokenizer):
        response, logits, start_pos = self.answer_question(question, answer, model, tokenizer)
        str_response = tokenizer.decode(response, skip_special_tokens=True)
        
        return response, str_response, logits, start_pos


    def get_start_end_layer(self, model):
        if "llama" in self.model_name:
            layer_count = model.model.layers
        elif "falcon" in self.model_name:
            layer_count = model.transformer.h
        else:
            layer_count = model.model.decoder.layers
        layer_st = 0 if SnyderEtAlEarlyDetection.layer_number == -1 else SnyderEtAlEarlyDetection.layer_number
        layer_en = len(layer_count) if SnyderEtAlEarlyDetection.layer_number == -1 else SnyderEtAlEarlyDetection.layer_number + 1
        return layer_st, layer_en


    def collect_fully_connected(self, token_pos, layer_start, layer_end):
        layer_name = SnyderEtAlEarlyDetection.model_repos[self.llm_name][1][2:].split(SnyderEtAlEarlyDetection.coll_str)
        first_activation = np.stack([SnyderEtAlEarlyDetection.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([SnyderEtAlEarlyDetection.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def collect_attention(self, token_pos, layer_start, layer_end):
        layer_name = SnyderEtAlEarlyDetection.model_repos[self.llm_name][2][2:].split(SnyderEtAlEarlyDetection.coll_str)
        first_activation = np.stack([SnyderEtAlEarlyDetection.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([SnyderEtAlEarlyDetection.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        norm = torch.norm(attributes, dim=1)
        attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
        
        return attributes


    def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
                -> torch.Tensor:
        output = model(inputs_embeds=input_, **extra_forward_args)
        return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


    def get_embedder(self, model):
        if "falcon" in self.llm_name:
            return model.transformer.word_embeddings
        elif "opt" in self.llm_name:
            return model.model.decoder.embed_tokens
        elif "llama" in self.llm_name:
            return model.model.embed_tokens
        else:
            raise ValueError(f"Unknown model {self.llm_name}")


    def get_ig(prompt, forward_func, tokenizer, embedder, model):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        prediction_id = SnyderEtAlEarlyDetection.get_next_token(input_ids, model).squeeze()[-1].argmax()
        encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
        ig = IntegratedGradients(forward_func=forward_func)
        attributes = SnyderEtAlEarlyDetection.normalize_attributes(
            ig.attribute(
                encoder_input_embeds,
                target=prediction_id,
                n_steps=SnyderEtAlEarlyDetection.ig_steps,
                internal_batch_size=SnyderEtAlEarlyDetection.internal_batch_size
            )
        ).detach().cpu().numpy()
        return attributes


    def compute_and_save_results(self):
        question_asker = self.answer_trivia
        
        forward_func = partial(SnyderEtAlEarlyDetection.model_forward, model=self.llm, extra_forward_args={})
        embedder = self.get_embedder(self.llm)

        # Prepare to save the internal states
        for name, module in self.llm.named_modules():
            if re.match(f'{SnyderEtAlEarlyDetection.model_repos[self.model_name][1]}$', name):
                SnyderEtAlEarlyDetection.fully_connected_forward_handles[name] = module.register_forward_hook(
                    partial(SnyderEtAlEarlyDetection.save_fully_connected_hidden, name))
            if re.match(f'{SnyderEtAlEarlyDetection.model_repos[self.model_name][2]}$', name):
                SnyderEtAlEarlyDetection.attention_forward_handles[name] = module.register_forward_hook(partial(SnyderEtAlEarlyDetection.save_attention_hidden, name))

        # Generate results
        results = defaultdict(list)
        for idx in tqdm(range(len(self.dataset))):
            SnyderEtAlEarlyDetection.fully_connected_hidden_layers.clear()
            SnyderEtAlEarlyDetection.attention_hidden_layers.clear()

            question, answer, instance_id = self.dataset[idx]
            response, str_response, logits, start_pos = question_asker(question, answer, self.llm, self.tokenizer)
            layer_start, layer_end = self.get_start_end_layer(self.llm)
            first_fully_connected, final_fully_connected = self.collect_fully_connected(start_pos, layer_start, layer_end)
            first_attention, final_attention = self.collect_attention(start_pos, layer_start, layer_end)
            attributes_first = SnyderEtAlEarlyDetection.get_ig(question, forward_func, self.tokenizer, embedder, self.llm)

            results['instance_id'].append(instance_id)
            results['question'].append(question)
            results['answers'].append(answer)
            results['response'].append(response)
            results['str_response'].append(str_response)
            results['logits'].append(logits.to(torch.float32).cpu().numpy())
            results['start_pos'].append(start_pos)
            results["fully"].append(first_fully_connected)
            results['final_fully_connected'].append(final_fully_connected)
            results["attn"].append(first_attention)
            results['final_attention'].append(final_attention)
            results["attribution"].append(attributes_first)
        
        self.results_pkl_path = os.path.join(self.project_dir, self.CACHE_DIR_NAME, f"{self.llm_name}_{self.dataset_name}_start-{SnyderEtAlEarlyDetection.start}_end-{SnyderEtAlEarlyDetection.end}_{datetime.now().month}_{datetime.now().day}.pickle")
        
        with open(self.results_pkl_path, "wb") as outfile:
            outfile.write(pickle.dumps(results))


    @staticmethod
    def compute_all_metrics(preds, data_name):
        # Convert preds to a df
        preds_df = pd.DataFrame(preds)
        
        # Compute metrics at dataset level
        all_preds = preds_df["prediction"].tolist()

        if data_name == SnyderEtAlEarlyDetection.DEFAULT_DATASET:
            labels = [1.] * len(all_preds)  # All instances are hallucinations
        else:
            labels = preds_df["label"].tolist()

        metrics = SnyderEtAlEarlyDetection.compute_metrics(all_preds, labels)

        # Compute metrics for each language - Only MashRoom has multiple languages
        if data_name == SnyderEtAlEarlyDetection.DEFAULT_DATASET:
            langs = preds_df["lang"].unique()
            for lang in langs:
                lang_preds = preds_df[preds_df["lang"] == lang]["prediction"].tolist()
                labels = [1.] * len(lang_preds)  # All instances are hallucinations
                lang_metrics = SnyderEtAlEarlyDetection.compute_metrics(lang_preds, labels)
                metrics[lang] = lang_metrics["ACC"]

        return metrics


    @staticmethod
    def compute_metrics(preds, labels):
        correct = sum([1 for p, l in zip(preds, labels) if p == l])

        AUC = roc_auc_score(labels, preds)
        precision, recall, thresholds = precision_recall_curve(labels, preds)
        AUPRC = auc(recall, precision)
        ACC = correct / len(labels)

        return {"ACC": ACC, "AUC": AUC, "AUPRC": AUPRC}
    

    # -------------
    # Utility Methods
    # -------------
    def _get_task_name(self, label):
        return self.LABELS[label]


    def _create_folders_if_not_exists(self, label=1):
        model_name = self.llm_name.split("/")[-1]

        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)

        task = self._get_task_name(label=label)
        print(f"Task: {task}")

        self.hidden_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_hidden", task)
        self.mlp_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_mlp", task)
        self.attn_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_attn", task)

        self.generation_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "generations", task)
        self.logits_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "logits", task)
        
        for sd in [self.hidden_save_dir, self.mlp_save_dir, self.attn_save_dir, self.generation_save_dir, self.logits_save_dir]:
            print(f"Creating directory: {sd}")
            if not os.path.exists(sd):
                os.makedirs(sd)

        print("\n\n")

    
    def _save_metrics(self, metrics, target, data_name, llm_name):
        metrics_path = os.path.join(self.project_dir, self.RESULTS_DIR, llm_name, data_name, target, f"metrics_layer{self.kc_layer}.json")

        if not os.path.exists(os.path.dirname(metrics_path)):
            os.makedirs(os.path.dirname(metrics_path))
        else:
            # If the file already exists, load existing metrics and add the new ones
            if os.path.exists(metrics_path):
                existing_metrics = json.load(open(metrics_path, "r"))
                metrics.update(existing_metrics)

        json.dump(metrics, open(metrics_path, "w"), indent=4)
        
        print(f"\t -> Metrics saved to {metrics_path}")
