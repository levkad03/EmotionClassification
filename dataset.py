import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.labels = {label: idx for idx, label in enumerate(data["emotion"].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["sentence"]
        tokens = self.tokenizer(text)
        indices = [
            self.vocab[token] if token in self.vocab else self.vocab["<unk>"]
            for token in tokens
        ]

        label = self.labels[self.data.iloc[idx]["emotion"]]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )
