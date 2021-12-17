import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

class MyEmbedding:
    def __init__(self, vocab="vocab/small_vocab.txt"):
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
    
    def embed(self, batch):
        batch_embedding, batch_mask, batch_indices = [], [], []
        max_len = max(len(words) for words in batch)
        for words in batch:
            indices = [self.wordToIndex[word] for word in words]
            indices += [2] * (max_len - len(words))             # 2 is [SEP]
            batch_mask.append(torch.IntTensor(indices) != 2)    # 2 is [SEP]
            batch_embedding.append(self.embedding(torch.IntTensor(indices)))
            batch_indices.append(torch.IntTensor(indices))
        batch_embedding = torch.stack(batch_embedding, dim=1)       # (max_length, batch_size, embed_dim)
        batch_mask = torch.stack(batch_mask, dim=1).unsqueeze(-1)   # (max_length, batch_size, 1)
        batch_indices = torch.stack(batch_indices, dim=1)           # (max_length, batch_size)
        return batch_embedding, batch_mask, batch_indices

    def vocabSize(self):
        return len(self.vocabs)
    
    def getIndex(self, word):
        return self.wordToIndex[word]
    
    def getWord(self, index):
        return self.vocabs[index]
