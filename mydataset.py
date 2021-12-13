import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
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
    def __init__(self, path, vocab="vocab/vocab.txt"):
        # Load pretrained word vectors
        logging.info("Loading word vectors ...")
        embeddings = []
        self.vocabs = []
        self.wordToIndex = defaultdict(int)
        with open(vocab, "r", encoding="utf8") as f:
            for i, line in tqdm(enumerate(f)):
                values = line.split()
                self.vocabs.append(values[0])
                self.wordToIndex[values[0]] = i
                vector = list(map(float, values[1:]))
                embeddings.append(torch.Tensor(vector))
        self.embedding = nn.Embedding.from_pretrained(torch.stack(embeddings))
        # Load data files
        logging.info("Loading data files ...")
        self.data = []
        with open(path, "r", encoding="utf8") as f:
            for line in tqdm(f):
                keys, values, src, tgt = line.split("\t\t")
                self.data.append({
                    "keys": [self.embed(key) for key in keys.split("\t")],
                    "values": [self.embed(value) for value in values.split("\t")],
                    "src": self.embed(src),
                    "tgt": self.embed(tgt)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def embed(self, sentence):
        wordIndices = [self.wordToIndex[word] for word in sentence.split()]
        return self.embedding(torch.IntTensor(wordIndices))

