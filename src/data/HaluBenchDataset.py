from datasets import load_dataset
from torch.utils.data import Dataset
from src.model.utils import get_weight_dir

REPO_NAME = "PatronusAI/HaluBench"
LABEL_MAPPING = {
    0: "PASS",
    1: "FAIL"
}

class HaluBenchDataset(Dataset):
    def __init__(self, label=0, use_local=False):
        if not use_local:
            self.data = load_dataset(REPO_NAME)['train']      # Dummy split, it can be val or test, too
        else:
            local_model_path = get_weight_dir(REPO_NAME, repo_type="datasets")
            self.data = load_dataset("parquet", data_dir=local_model_path)['train']      # Dummy split, it can be val or test, too
            print(self.data)

        self.dataset = self.remove_instances_by_label(label)
        self.dataset = self.dataset.shuffle(seed=42)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        id = self.dataset[idx]['id']

        question = self.dataset[idx]['question'] + "\n[Further Knowledge]\n" + self.dataset[idx]['passage']
        answer = self.dataset[idx]['answer']

        return question, answer, id
    

    def get_language_by_instance_id(self, id):
        matches = self.dataset.filter(lambda x: x["id"] == id)

        if len(matches) == 0:
            raise ValueError(f"Instance ID {id} non trovato.")
        
        return matches[0]["lang"]

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("id")

        self.dataset = self.dataset.add_column("id", instance_ids)

        return self.dataset


    def save_dataset_as_jsonl(self, output_path):
        self.dataset.to_json(output_path, lines=True, orient="records")


    def remove_instances_by_label(self, label):
        self.dataset = self.dataset.filter(lambda x: x['label'] != LABEL_MAPPING[label])
        return self.dataset