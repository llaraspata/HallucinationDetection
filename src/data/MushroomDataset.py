from datasets import load_dataset
from torch.utils.data import Dataset


class MushroomDataset(Dataset):
    def __init__(self, data_path, data_extension="json"):
        self.dataset = load_dataset(data_extension, data_files=data_path)['train']      # Dummy split, it can be val or test, too
        self.dataset = self.dataset.shuffle(seed=42)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        question = self.dataset[idx]['model_input']
        answer = self.dataset[idx]['model_output_text']
        lang = self.dataset[idx]['lang']
        return question, answer, lang
