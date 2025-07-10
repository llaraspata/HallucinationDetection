from datasets import load_dataset
from torch.utils.data import Dataset


class MushroomDataset(Dataset):
    def __init__(self, data_path, data_extension="json", recreate_ids=False):
        self.dataset = load_dataset(data_extension, data_files=data_path)['train']      # Dummy split, it can be val or test, too

        if ('instance_id' not in self.dataset.column_names) or recreate_ids:
            self.dataset = self.create_instance_ids()
            self.save_dataset_as_jsonl(output_path=data_path)
        
        self.dataset = self.dataset.shuffle(seed=42)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        question = self.dataset[idx]['model_input']
        answer = self.dataset[idx]['model_output_text']
        id = self.dataset[idx]['instance_id']
        return question, answer, id
    

    def get_language_by_instance_id(self, instance_id):
        matches = self.dataset.filter(lambda x: x["instance_id"] == instance_id)

        if len(matches) == 0:
            raise ValueError(f"Instance ID {instance_id} non trovato.")
        
        return matches[0]["lang"]

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("instance_id")

        self.dataset = self.dataset.add_column("instance_id", instance_ids)

        return self.dataset



    def save_dataset_as_jsonl(self, output_path):
        self.dataset.to_json(output_path, lines=True, orient="records")
