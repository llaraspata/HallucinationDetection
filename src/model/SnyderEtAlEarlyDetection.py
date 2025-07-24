import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import src.model.utils as ut
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # Batch processing
    BATCH_SIZE = 8  # Number of samples to process in parallel

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
    def __init__(self, project_dir, batch_size=None):
        self.project_dir = project_dir
        if batch_size is not None:
            self.BATCH_SIZE = batch_size
        else:
            # Auto-determine optimal batch size
            self.BATCH_SIZE = self.get_optimal_batch_size()

    
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


    def load_llm(self, llm_name, use_local=False):
        print("--"*50)
        print(f"Loading LLM {llm_name}")
        print("--"*50)
        self.llm_name = f'{self.model_repos[llm_name][0]}/{llm_name}'

        if use_local:
            model_local_path = ut.get_weight_dir(self.llm_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_local_path, local_files_only=True, token=True)
            self.llm = AutoModelForCausalLM.from_pretrained(model_local_path, local_files_only=True,
                                                            quantization_config=ut.create_bnb_config(),
                                                            device_map={'':PartialState().process_index},
                                                            torch_dtype=torch.bfloat16)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, token=True)
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name, 
                                                            quantization_config=ut.create_bnb_config(),
                                                            device_map={'':PartialState().process_index},
                                                            torch_dtype=torch.bfloat16)
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
    def predict_llm(self, llm_name, data_name=DEFAULT_DATASET, label=1, use_local=False, batch_size=None, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
            
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)
        self.load_llm(llm_name, use_local=use_local)

        print("--"*50)
        print("Hallucination Detection - Saving LLM's activations")
        print(f"Batch size: {batch_size}")
        print("--"*50)
        
        print("\n0. Prepare folders")
        self._create_folders_if_not_exists(label=label)
    
        print(f"\n1. Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.compute_and_save_results(batch_size=batch_size)
        
        print("--"*50)

    
    @torch.no_grad()
    def predict_kc(self, llm_name, activation="logits", layer=0, data_name=DEFAULT_DATASET, use_local=False, label=1):
        self.load_hal_probing(activation=activation, layer=layer)
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)

        print("--"*50)
        print("Hallucination Detection - Saving Hallucination Probing Predictions")
        print(f"Label: {self.LABELS[label]}")
        print("--"*50)
        model_name = llm_name.split("/")[-1]

        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, model_name, self.dataset_name)
        
        activations_path = os.path.join(self.project_dir, self.CACHE_DIR_NAME, model_name, self.dataset_name, self.LABELS[label])
        act_files = os.listdir(activations_path)
        act_files = [f for f in act_files if f.endswith(".pickle")]

        preds = []

        if activation == "logits" or activation == "attribution":
            preds_filename = self.PREDICTIONS_FILE_NAME

            for instance_file in tqdm(act_files, desc=f"Processing {activation}"):
                with open(os.path.join(activations_path, instance_file), "rb") as f:
                    activation_data = pickle.load(f)

                instance_id = activation_data['instance_id']

                pred = {
                    "instance_id": instance_id,
                    "lang": self.dataset.get_language_by_instance_id(instance_id),
                    "prediction": self.probing_model.predict(activation_data[activation]).item(),
                    "label": label
                }

                preds.append(pred)

        else:
            preds_filename = self.PREDICTIONS_LAYER_FILE_NAME.format(layer=layer)
            
            for instance_file in tqdm(act_files, desc=f"Processing {activation} layer {layer}"):
                with open(os.path.join(activations_path, instance_file), "rb") as f:
                    activation_data = pickle.load(f)

                instance_id = activation_data['instance_id'][layer]

                pred = {
                    "instance_id": instance_id,
                    "lang": self.dataset.get_language_by_instance_id(instance_id),
                    "prediction": self.probing_model.predict(activation_data[activation][layer]).item(),
                    "label": label
                }

                preds.append(pred)

        path_to_save = os.path.join(result_path, activation, preds_filename)

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
    def compute_and_save_results(self, batch_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
            
        question_asker = self.answer_trivia_batch
        model_name = self.llm_name.split("/")[-1]
        
        forward_func = partial(SnyderEtAlEarlyDetection.model_forward, model=self.llm, extra_forward_args={})
        embedder = self.get_embedder(self.llm)

        # Prepare to save the internal states
        for name, module in self.llm.named_modules():
            if re.match(f'{self.model_repos[model_name][1]}$', name):
                self.fully_connected_forward_handles[name] = module.register_forward_hook(
                    partial(SnyderEtAlEarlyDetection.save_fully_connected_hidden, name))
            if re.match(f'{self.model_repos[model_name][2]}$', name):
                self.attention_forward_handles[name] = module.register_forward_hook(partial(SnyderEtAlEarlyDetection.save_attention_hidden, name))

        # Generate results in batches
        dataset_size = len(self.dataset)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        print(f"Processing {dataset_size} samples in {num_batches} batches of size {batch_size}")
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            current_batch_size = end_idx - start_idx
            
            print(f"Batch {batch_idx + 1}/{num_batches}: processing samples {start_idx} to {end_idx-1} ({current_batch_size} samples)")
            
            # Clear accumulated activations
            self.fully_connected_hidden_layers.clear()
            self.attention_hidden_layers.clear()
            
            # Prepare batch data
            batch_questions = []
            batch_answers = []
            batch_instance_ids = []
            
            for idx in range(start_idx, end_idx):
                question, answer, instance_id = self.dataset[idx]
                batch_questions.append(question)
                batch_answers.append(answer)
                batch_instance_ids.append(instance_id)
            
            # Process batch
            batch_results = question_asker(batch_questions, batch_answers, self.llm, self.tokenizer)
            layer_start, layer_end = self.get_start_end_layer(self.llm)
            
            # Process each sample in the batch
            for i, (responses, str_responses, logits, start_pos) in enumerate(batch_results):
                instance_id = batch_instance_ids[i]
                question = batch_questions[i]
                answer = batch_answers[i]
                
                first_fully_connected, final_fully_connected = self.collect_fully_connected_batch(start_pos, layer_start, layer_end, i)
                first_attention, final_attention = self.collect_attention_batch(start_pos, layer_start, layer_end, i)
                attributes_first = SnyderEtAlEarlyDetection.get_ig(question, forward_func, self.tokenizer, embedder, self.llm)
                
                result = {
                    'instance_id': instance_id,
                    'question': question,
                    'answers': answer,
                    'response': responses,
                    'str_response': str_responses,
                    'logits': logits.to(torch.float32).cpu().numpy(),
                    'start_pos': start_pos,
                    "fully": first_fully_connected,
                    'final_fully_connected': final_fully_connected,
                    "attn": first_attention,
                    'final_attention': final_attention,
                    "attribution": attributes_first
                }

                result_pkl_path = os.path.join(self.project_dir, self.CACHE_DIR_NAME, model_name, self.dataset_name, self.LABELS[self.label], f"data_instance{instance_id}.pickle")
                os.makedirs(os.path.dirname(result_pkl_path), exist_ok=True)
                
                with open(result_pkl_path, "wb") as outfile:
                    outfile.write(pickle.dumps(result))

                del first_fully_connected, final_fully_connected, first_attention, final_attention, attributes_first
            
            # Clean up batch results
            del batch_results
            torch.cuda.empty_cache()

        # Deallocate LLM and tokenizer + clear cache
        self.llm = None
        self.tokenizer = None
        torch.cuda.empty_cache()


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
    @staticmethod
    def get_optimal_batch_size(available_memory_gb=None):
        """Calculate optimal batch size based on available GPU memory"""
        if available_memory_gb is None:
            if torch.cuda.is_available():
                # Get available GPU memory in GB
                available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                available_memory_gb = 8  # Default for CPU
        
        # Rough heuristic: larger models need smaller batch sizes
        if available_memory_gb >= 32:
            return 16
        elif available_memory_gb >= 16:
            return 8
        elif available_memory_gb >= 8:
            return 4
        else:
            return 2


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
        input_ids = tokenizer(model_input, return_tensors='pt', truncation=True ).input_ids.to(model.device)
        response, logits = self.generate_response(input_ids, model, max_length=max_length, pbar=pbar)
        return response, logits, input_ids.shape[-1]


    def answer_trivia(self, question, answer, model, tokenizer):
        response, logits, start_pos = self.answer_question(question, answer, model, tokenizer)
        str_response = tokenizer.decode(response, skip_special_tokens=True)
        
        return response, str_response, logits, start_pos


    def answer_trivia_batch(self, questions, answers, model, tokenizer):
        """Process multiple questions in batch for better efficiency"""
        batch_results = []
        
        # Prepare batch inputs
        batch_prompts = []
        for question, answer in zip(questions, answers):
            model_input = prompt.format(question=question, answer=answer)
            batch_prompts.append(model_input)
        
        # Tokenize batch
        batch_encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        batch_input_ids = batch_encoded.input_ids.to(model.device)
        batch_attention_mask = batch_encoded.attention_mask.to(model.device)
        
        # Get start positions for each sample in batch
        start_positions = batch_input_ids.shape[-1]
        
        # Process each sample individually due to generation complexity
        # Full batch generation would require significant refactoring of generation logic
        for i in range(len(questions)):
            input_ids = batch_input_ids[i:i+1]  # Keep batch dimension
            response, logits = self.generate_response(input_ids, model, max_length=self.MAX_NEW_TOKENS)
            str_response = tokenizer.decode(response, skip_special_tokens=True)
            batch_results.append((response, str_response, logits, start_positions))
        
        return batch_results


    def answer_trivia_batch_optimized(self, questions, answers, model, tokenizer):
        """Optimized batch processing - experimental"""
        # This is a more experimental approach that could be developed further
        # for true parallel generation across the batch
        batch_results = []
        
        # For now, process sequentially but with better memory management
        for question, answer in zip(questions, answers):
            response, str_response, logits, start_pos = self.answer_trivia(question, answer, model, tokenizer)
            batch_results.append((response, str_response, logits, start_pos))
        
        return batch_results


    def get_start_end_layer(self, model):
        if "llama" in self.llm_name:
            layer_count = model.model.layers
        elif "falcon" in self.llm_name:
            layer_count = model.transformer.h
        else:
            layer_count = model.model.decoder.layers
        layer_st = 0 if self.layer_number == -1 else self.layer_number
        layer_en = len(layer_count) if self.layer_number == -1 else self.layer_number + 1
        return layer_st, layer_en


    def collect_fully_connected(self, token_pos, layer_start, layer_end):
        model_name = self.llm_name.split("/")[-1]

        layer_name = self.model_repos[model_name][1][2:].split(self.coll_str)
        first_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def collect_fully_connected_batch(self, token_pos, layer_start, layer_end, batch_idx):
        """Collect fully connected activations for a specific item in the batch"""
        model_name = self.llm_name.split("/")[-1]

        layer_name = self.model_repos[model_name][1][2:].split(self.coll_str)
        
        # Get activations for the specific batch index
        first_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][batch_idx][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][batch_idx][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def collect_attention(self, token_pos, layer_start, layer_end):
        model_name = self.llm_name.split("/")[-1]

        layer_name = self.model_repos[model_name][2][2:].split(self.coll_str)
        first_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                    for i in range(layer_start, layer_end)])
        return first_activation, final_activation


    def collect_attention_batch(self, token_pos, layer_start, layer_end, batch_idx):
        """Collect attention activations for a specific item in the batch"""
        model_name = self.llm_name.split("/")[-1]

        layer_name = self.model_repos[model_name][2][2:].split(self.coll_str)
        
        # Get activations for the specific batch index
        first_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][batch_idx][token_pos] \
                                    for i in range(layer_start, layer_end)])
        final_activation = np.stack([self.attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][batch_idx][-1] \
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
