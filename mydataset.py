import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

def collate_fn(batch):
    keys, values, src, tgt = [], [], [], []
    for each in batch:
        keys.append(each["keys"])
        values.append(each["values"])
        src.append(each["src"])
        tgt.append(each["tgt"])
    return keys, values, src, tgt

class MyDataset(Dataset):
    def __init__(self, path):
        # Load data files
        self.data = []
        with open(path, "r", encoding="utf8") as f:
            for line in tqdm(f):
                keys, values, src, tgt = line.split("\t\t")
                self.data.append({
                    "keys": [key.split() for key in keys.split("\t")],
                    "values": [value.split() for value in values.split("\t")],
                    "src": src.split(),
                    "tgt": tgt.split()
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

