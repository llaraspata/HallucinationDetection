import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from src.data.MushroomDataset import MushroomDataset


PROJECT_ROOT = os.getcwd()
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FINAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

VAL_DIR = os.path.join(RAW_DATA_DIR, "val")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
MERGED_PATH = os.path.join(FINAL_DATA_DIR, "labeled.jsonl")

COLUMNS_TO_KEEP = [
    "model_id",
    "model_input",
    "model_output_text",
    "lang",
]


def main():
    print("--" * 50)
    print("Join labeled dataset to unique JSONL file")
    print("--" * 50)

    dir_list = [VAL_DIR, TEST_DIR]
    dataset = None

    # Join all validation and test JSONL files into a single dataset
    for dir_path in dir_list:
        print(f"\n\nProcessing directory: {dir_path}")

        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist. Please check the path.")
            return
        
        dir_files = os.listdir(dir_path)
        
        for file_name in dir_files:
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(dir_path, file_name)
                
                ds = load_dataset('json', data_files=file_path)['train']
                
                # Keep only the relevant columns
                ds = ds.remove_columns([col for col in ds.column_names if col not in COLUMNS_TO_KEEP])
                
                dataset = concatenate_datasets([dataset, ds]) if dataset is not None else ds
    
    # Save the merged dataset to a JSONL file
    dataset.to_json(MERGED_PATH, lines=True, orient="records")

    # Load the merged dataset and recreate instance IDs
    labeled_ds = MushroomDataset(MERGED_PATH, recreate_ids=True)

    print("--" * 50)
    

if __name__ == '__main__':
    main()
