from datasets import load_dataset
from torch.utils.data import Dataset
from src.model.utils import get_weight_dir

REPO_NAME = "PatronusAI/HaluBench"
LABEL_MAPPING = {
    0: "PASS",
    1: "FAIL"
}

class HaluBenchDataset(Dataset):
    def __init__(self, label=0, recreate_ids=True, use_local=False):
        if not use_local:
            self.dataset = load_dataset(REPO_NAME)['test']
        else:
            local_model_path = get_weight_dir(REPO_NAME, repo_type="datasets")
            self.dataset = load_dataset("parquet", data_dir=local_model_path)['test']

        if ('instance_id' not in self.dataset.column_names) or recreate_ids:
            self.dataset = self.create_instance_ids()

        self.dataset = self.remove_instances_by_label(label)
        self.dataset = self.dataset.shuffle(seed=42)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        id = self.dataset[idx]['instance_id']

        question = self.dataset[idx]['question'] + "\n[Further Knowledge]\n" + self.dataset[idx]['passage']
        answer = self.dataset[idx]['answer']

        return question, answer, id
    

    def get_language_by_instance_id(self, instance_id):
        return "EN"  # HaluEval is in English, so we return "EN" directly

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("instance_id")

        self.dataset = self.dataset.add_column("instance_id", instance_ids)

        return self.dataset


    def save_dataset_as_jsonl(self, output_path):
        self.dataset.to_json(output_path, lines=True, orient="records")


    def remove_instances_by_label(self, label):
        self.dataset = self.dataset.filter(lambda x: x['label'] != LABEL_MAPPING[label])
        return self.dataset