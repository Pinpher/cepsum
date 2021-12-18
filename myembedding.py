import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

class MyEmbedding:
    def __init__(self, device, vocab="vocab/vocab.txt"):
        self.device = device
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
        self.embedding = nn.Embedding.from_pretrained(torch.stack(embeddings), freeze=False)
        self.sepId = self.wordToIndex['[SEP]']
    
    def embed(self, batch):
        batch_embedding, batch_mask, batch_indices = [], [], []
        max_len = max(len(words) for words in batch)
        for words in batch:
            indices = [self.wordToIndex[word] for word in words]
            indices += [self.sepId] * (max_len - len(words))            
            batch_mask.append(torch.IntTensor(indices) != self.sepId)
            batch_embedding.append(self.embedding(torch.IntTensor(indices)))
            batch_indices.append(torch.IntTensor(indices))
        batch_embedding = torch.stack(batch_embedding, dim=1).to(self.device)       # (max_length, batch_size, embed_dim)
        batch_mask = torch.stack(batch_mask, dim=1).unsqueeze(-1).to(self.device)   # (max_length, batch_size, 1)
        batch_indices = torch.stack(batch_indices, dim=1).to(self.device)           # (max_length, batch_size)
        return batch_embedding, batch_mask, batch_indices

    def vocabSize(self):
        return len(self.vocabs)
    
    def getWord(self, index):
        return self.vocabs[index]
