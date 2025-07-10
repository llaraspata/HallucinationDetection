import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

import ast
import json
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_PKL = os.path.join(DATA_PATH, "train.pkl")
VAL_PKL = os.path.join(DATA_PATH, "val.pkl")
TEST_PKL = os.path.join(DATA_PATH, "test.pkl")

TRAIN_PATH = os.path.join(DATA_PATH, "train.jsonl")
VAL_PATH = os.path.join(DATA_PATH, "val.jsonl")
TEST_PATH = os.path.join(DATA_PATH, "test.jsonl")

PATHS = {
    "train": [TRAIN_PKL, TRAIN_PATH],
    "val": [VAL_PKL, VAL_PATH],
    "test": [TEST_PKL, TEST_PATH],
}

COLUMNS_TO_CHECK = ["model_input_tokens", "model_output_logits"]


def main():
    print("--" * 50)
    print("Dataset conversion to JSONL format")
    print("--" * 50)

    for split, [pkl_path, result_path] in PATHS.items():
        print(f"\nProcessing {split} dataset...")
        dataset = pd.read_pickle(pkl_path)
        convert_to_jsonl(dataset, result_path)

    print("--" * 50)
    

# -----------------
# Helper functions
# -----------------
def convert_to_jsonl(df: pd.DataFrame, result_path: str) -> list:
    """
    Converts a DataFrame to a list of dictionaries suitable for JSONL format.

    Args:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        list: List of dictionaries representing the DataFrame rows.
    """
    for col in COLUMNS_TO_CHECK:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    dict_list = df.to_dict(orient='records')

    write_jsonl(dict_list, result_path)


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []
    

def write_jsonl(data: list, filepath: str):
    """
    Writes a list of JSON/dict objects to a .jsonl file.

    Args:
        data (list): List of dictionaries (JSON-like objects).
        filepath (str): Path to the output .jsonl file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Writing JSONL"):
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    f.close()

    print(f"\t -> Data successfully written to {filepath}")


if __name__ == '__main__':
    main()