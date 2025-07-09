import os
import json
import torch
from tqdm import tqdm
import src.model.utils as ut
from src.model.KCProbing import KCProbing
from src.data.MushroomDataset import MushroomDataset
from src.model.InspectOutputContext import InspectOutputContext
from src.model.prompts import PROMPT_QA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, precision_recall_curve

class HallucinationDetection:
    # -------------
    # Constants
    # -------------
    TARGET_LAYERS = list(range(14, 17))     # Upper bound excluded
    MAX_NEW_TOKENS = 100
    DATASET_NAME = "mushroom"
    CACHE_DIR_NAME = "activation_cache"
    TASK = "hallucination_detection"
    ACTIVATION_TARGET = ["hidden"]          # Here only hidden, but in the referece study were: hidden, mlp, attn
    PREDICTION_DIR = "predictions"
    PREDICTIONS_FILE_NAME = "kc_predictions_layer{layer}.jsonl"
    LABELS = {
        0: "Not Hallucination",
        1: "Hallucination"
    }

    # -------------
    # Constructor
    # -------------
    def __init__(self, project_dir, llm_name, kc_layer, use_local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        if kc_layer not in self.TARGET_LAYERS:
            raise ValueError(f"KC layer must be one of {self.TARGET_LAYERS}, but got {kc_layer}")
        print("--"*50)
        print("Init")
        print("--"*50)

        self.project_dir = project_dir

        print(f"\n 1. Loading Knowledge-Conflict Probing model for layer {kc_layer}")
        self.kc_layer = kc_layer
        self.kc_model = KCProbing(project_dir, layer=kc_layer)

        print(f"\n 2. Loading LLM {llm_name}")
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.llm = ut.load_llm(llm_name, ut.create_bnb_config(), local=use_local, dtype=dtype, use_device_map=use_device_map, use_flash_attention=use_flash_attn)
        
        print(f"\n 3. Loading {self.DATASET_NAME} dataset")
        val_path = os.path.join(project_dir, "data", "processed", "val.jsonl")
        self.dataset = MushroomDataset(data_path=val_path)
        print("--"*50)


    # -------------
    # Main Methods
    # -------------
    @torch.no_grad()
    def predict(self):
        print("--"*50)
        print("Hallucination Detection")
        print("--"*50)
        
        print("\n0. Prepare folders")
        self._create_folders_if_not_exists()
       
        print(f"\n1. Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.save_avtivations()

        print("\n2. Running KC Probing")
        self.predict_kc()
        
        print("--"*50)


    def eval(self):
        print("--"*50)
        print("Hallucination Detection Evaluation")
        print("--"*50)
        
        print(f"\n1. Load predictions for layer {self.kc_layer}")
        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Predictions file not found: {result_path}")
        
        preds = json.load(open(result_path, "r"))

        labels = [1.] * len(preds)  # All instances are hallucinations

        print("\n2. Compute metrics")
        metrics = HallucinationDetection.compute_metrics(preds, labels)

        print("\n3. Save results")
        self._save_metrics(metrics)        
        
        print("--"*50)


    # -------------
    # Public Methods
    # -------------
    def save_avtivations(self):
        module_names = [f'model.layers.{idx}' for idx in self.TARGET_LAYERS]

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations"):
            question, answer, lang = self.dataset[idx]

            model_input = PROMPT_QA.format(question=question, answer=answer)
            tokens = self.tokenizer(model_input, return_tensors="pt")
            attention_mask = tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else None

            with InspectOutputContext(self.llm, module_names, save_generation=True, save_dir=self.generation_save_dir) as inspect:
                output = self.llm.generate(
                    input_ids=tokens["input_ids"].to("cuda"),
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                ut.save_generation_output(generated_text, model_input, idx, self.generation_save_dir)
                
                if hasattr(output, 'scores') and output.scores:
                    logits = torch.stack(output.scores, dim=1)  # [batch, seq_len, vocab_size]
                    ut.save_model_logits(logits, idx, self.logits_save_dir)
                
            for module, ac in inspect.catcher.items():
                # ac: [batch_size, sequence_length, hidden_dim]
                ac_last = ac[0, -1].float()
                layer_idx = int(module.split(".")[2])

                save_name = f"layer{layer_idx}-id{idx}.pt"
                save_path = os.path.join(self.hidden_save_dir, save_name)
                torch.save(ac_last, save_path)

        self.combine_activations()


    def combine_activations(self):
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        model_name = self.llm_name.split("/")[-1]
        
        for aa in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.DATASET_NAME, f"activation_{aa}", self.TASK)

            act_files = list(os.listdir(act_dir))
            act_files = [f for f in act_files if len(f.split("-")) == 2]

            act_files_layer_idx_instance_idx = [
                [act_f, ut.parse_layer_id_and_instance_id(os.path.basename(act_f))]
                for act_f in act_files
            ]

            # For each layer id (as key), the value contains a list of [activation file, instance id]
            layer_group_files = {lid: [] for lid in self.TARGET_LAYERS}
            for act_f, (layer_id, instance_id) in act_files_layer_idx_instance_idx:
                layer_group_files[layer_id].append([act_f, instance_id])
                        
            for layer_id in self.TARGET_LAYERS:
                # Sort the files for each layer by instance ID
                layer_group_files[layer_id] = sorted(layer_group_files[layer_id], key=lambda x: x[1])

                acts = []
                loaded_paths = []
                for idx, (act_f, instance_id) in enumerate(layer_group_files[layer_id]):
                    assert idx == instance_id
                    path_to_load = os.path.join(act_dir, act_f)
                    acts.append(torch.load(path_to_load))
                    loaded_paths.append(path_to_load)

                acts = torch.stack(acts)
                save_path = os.path.join(act_dir, f"layer{layer_id}_activations.pt")
                torch.save(acts, save_path)

                for p in loaded_paths:
                    os.remove(p)


    def predict_kc(self):
        activations = ut.load_activations(
            model_name=self.llm_name,
            data_name=self.DATASET_NAME,
            analyse_activation=self.ACTIVATION_TARGET[0],  # Only hidden activations
            activation_type=self.TASK,
            layer_idx=self.kc_layer,
            results_dir=os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        )

        preds = []
        for activation in activations:
            preds.append(self.kc_model.predict(activation).item())

        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        json.dump(preds, open(result_path, "w"), indent=4)
        
        print(f"\t -> Predictions saved to {result_path}")


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
    def _create_folders_if_not_exists(self):
        model_name = self.llm_name.split("/")[-1]

        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)

        self.hidden_save_dir = os.path.join(results_dir, model_name, self.DATASET_NAME, "activation_hidden", self.TASK)
        self.generation_save_dir = os.path.join(results_dir, model_name, self.DATASET_NAME, "generations", self.TASK)
        self.logits_save_dir = os.path.join(results_dir, model_name, self.DATASET_NAME, "logits", self.TASK)
        
        for sd in [self.hidden_save_dir, self.generation_save_dir, self.logits_save_dir]:
            if not os.path.exists(sd):
                os.makedirs(sd)

    
    def _save_metrics(self, metrics):
        metrics_path = os.path.join(self.project_dir, self.PREDICTION_DIR, f"metrics_layer{self.kc_layer}.json")

        if not os.path.exists(os.path.dirname(metrics_path)):
            os.makedirs(os.path.dirname(metrics_path))

        json.dump(metrics, open(metrics_path, "w"), indent=4)
        
        print(f"\t -> Metrics saved to {metrics_path}")
