import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """Language modeling dataset from pre-tokenized uint16 .bin files."""

    def __init__(self, bin_path: str, seq_length: int):
        data = np.fromfile(bin_path, dtype=np.uint16)
        n_seq = len(data) // seq_length
        self.data = data[: n_seq * seq_length].reshape(n_seq, seq_length)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]
