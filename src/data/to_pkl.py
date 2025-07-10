import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

import ast
import json
import pandas as pd


PROJECT_ROOT = os.getcwd()

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train")
VAL_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "val")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "test")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

COLUMNS_TO_CHECK = ["model_input_tokens", "model_output_logits"]


def main():
    print("----- TRAIN -----")
    train_df = load_whole_set(TRAIN_PATH)
    
    print("\n----- VAL -----")
    val_df = load_whole_set(VAL_PATH)

    print("\n----- TEST -----")
    test_df = load_whole_set(TEST_PATH)
    
    print("\n\nSaving train and validation sets to pickle!")
    train_df.to_pickle(os.path.join(DATA_PATH, "train.pkl"))
    val_df.to_pickle(os.path.join(DATA_PATH, "val.pkl"))
    test_df.to_pickle(os.path.join(DATA_PATH, "test.pkl"))

    print("----- END -----")


def load_whole_set(folder_path):
    result = pd.DataFrame()

    for element in os.listdir(folder_path):
        element_path = os.path.join(folder_path, element)
        
        if (not os.path.isdir(element_path)) and (element.endswith("jsonl")):
            print(f"\t - Loading {element}")

            df = load_data(element_path)
            result = pd.concat([result, df], ignore_index=True)

    return result


def load_data(data_path):
    dataset = []
    
    with open(data_path, 'r') as data:
        for line in data:
            dataset.append(json.loads(line))
    
    data.close()

    dataset_df = pd.DataFrame(dataset)

    for col in COLUMNS_TO_CHECK:
        if col in dataset_df.columns:
            dataset_df[col] = dataset_df[col].apply(safe_parse_list)

    return dataset_df


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


if __name__ == '__main__':
    main()