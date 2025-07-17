import os
import json
import torch
from pathlib import Path
from typing import Union, Any
from accelerate import PartialState
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from src.model.LlaMa import LlamaForCausalLM
from src.model.InspectOutputContext import InspectOutputContext



HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")



def get_weight_dir(
    model_ref: str,
    *,
    model_dir: Union[str, os.PathLike[Any]] = HF_DEFAULT_HOME,
    revision: str = "main",
    repo_type="models",
    subset=None,
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist or is not a directory."

    model_path = Path(os.path.join(model_dir, "hub", "--".join([repo_type, *model_ref.split("/")])))
    assert model_path.is_dir(), f"Model path {model_path} does not exist or is not a directory."
    
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir(), f"Weight directory {weight_dir} does not exist or is not a directory."

    if repo_type == "datasets":
        if subset is not None:
            weight_dir = weight_dir / subset
        else:
            # For datasets, we need to return the directory containing the dataset files
            weight_dir = weight_dir / "data"
    
    return weight_dir


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_llm(model_name, bnb_config, local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attention=False):
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "10000MB" for i in range(n_gpus)}
    attention = "flash_attention_2" if use_flash_attention else "eager"
    device_string = PartialState().process_index
    max_memory_config = max_memory if use_device_map else None

    if not local:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            attn_implementation=attention,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map={'':device_string},#device_map_config,       #{'':device_string},           #{"": 0},  # forces model to cuda:0 otherwise set as done in the else branch
            max_memory=max_memory_config
        )
    else:
        model_local_path = get_weight_dir(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_local_path,
            local_files_only=True,
            use_cache=False,
            attn_implementation=attention,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map={'':device_string}, #device_map_config,       #{'':device_string}, # dispatch efficiently the model on the available ressources
            max_memory=max_memory_config,
        )

    return model


def load_tokenizer(model_name, local=False):
    if not local:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    else:
        model_local_path = get_weight_dir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_local_path, local_files_only=True, token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_messages(system_prompt, user_prompt, k=0, sample_user_prompts=[], assistant_prompts=[], assistant=""):
    """
        Builds the messages to be sent to the LLM.
    """
    messages = []

    messages.append({ "role": "system", "content": system_prompt })

    for i in range(k):
        messages.append({ "role": "user", "content": sample_user_prompts[i][0] })
        messages.append({ "role": "assistant", "content": assistant_prompts[i][0] })

    messages.append({"role": "user", "content": user_prompt})

    if not (assistant == ""):
        messages.append({ "role": "assistant", "content": assistant })

    return messages


def save_generation_output(output_text, input_text, instance_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_data = {
        "instance_id": instance_id,
        "input": input_text,
        "generated_output": output_text,
        "timestamp": str(torch.cuda.current_stream().query())
    }
    
    save_path = os.path.join(save_dir, f"generation_{instance_id}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return save_path


def save_model_logits(logits, instance_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"logits_{instance_id}.pt")
    torch.save(logits.cpu(), save_path)

    return save_path


def parse_layer_id_and_instance_id(s):
    instance_idx = -1
    layer_idx = -1

    try:
        layer_s, id_s = s.split("-")
        layer_idx = int(layer_s[len("layer"):])
        instance_idx = int(id_s[len("id"):-len(".pt")])
    except Exception as e:
        print(s)
    return layer_idx, instance_idx


def load_activations(
        model_name="meta-llama/Meta-Llama-3-8B",
        data_name="demo",
        analyse_activation="hidden",
        activation_type="test",
        layer_idx=None,
        results_dir="cache_data",
    ) -> torch.Tensor:
    model_name = model_name.split("/")[-1]
    act_dir = os.path.join(results_dir, model_name, data_name, f"activation_{analyse_activation}", activation_type, f"layer{layer_idx}_activations.pt")
    
    ids_path = os.path.join(results_dir, model_name, data_name, f"activation_{analyse_activation}", activation_type, f"layer{layer_idx}_instance_ids.json")
    instance_ids = json.load(open(ids_path, "r"))

    return torch.load(act_dir, map_location="cuda"), instance_ids

