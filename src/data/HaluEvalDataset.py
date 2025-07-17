from datasets import load_dataset
from torch.utils.data import Dataset
from src.model.utils import get_weight_dir

REPO_NAME = "pminervini/HaluEval"

class HaluEvalDataset(Dataset):
    def __init__(self, label=0, recreate_ids=True, use_local=False):
        if not use_local:
            self.dataset = load_dataset(REPO_NAME, "dialogue")['train']      # Dummy split, it can be val or test, too
        else:
            local_model_path = get_weight_dir(REPO_NAME, repo_type="datasets", subset="dialogue")
            self.dataset = load_dataset("parquet", data_dir=local_model_path)['train']      # Dummy split, it can be val or test, too

        if ('instance_id' not in self.dataset.column_names) or recreate_ids:
            self.dataset = self.create_instance_ids()
        
        self.label = label
        self.dataset = self.dataset.shuffle(seed=42)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        id = self.dataset[idx]['instance_id']

        history = self.dataset[idx]['dialogue_history']
        knowledge = self.dataset[idx]['knowledge']

        question = history + "\n[Further Knowledge]\n" + knowledge

        if self.label == 0:
            answer = self.dataset[idx]['right_response']
        else:
            answer = self.dataset[idx]['hallucinated_response']

        return question, answer, id
    

    def get_language_by_instance_id(self, instance_id):
        return "EN"  # HaluEval is in English, so we return "EN" directly

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("instance_id")

        self.dataset = self.dataset.add_column("instance_id", instance_ids)

        return self.dataset
