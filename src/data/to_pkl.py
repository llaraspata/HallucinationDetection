import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

import json
import pandas as pd


PROJECT_ROOT = os.getcwd()

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train")
VAL_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "val")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")


def main():
    print("----- TRAIN -----")
    train_df = load_whole_set(TRAIN_PATH)
    
    
    print("\n----- VAL -----")
    val_df = load_whole_set(VAL_PATH)
    
    print("\n\nSaving train and validation sets to pickle!")
    train_df.to_pickle(os.path.join(DATA_PATH, "train.pkl"))
    val_df.to_pickle(os.path.join(DATA_PATH, "val.pkl"))

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

    return dataset_df


if __name__ == '__main__':
    main()