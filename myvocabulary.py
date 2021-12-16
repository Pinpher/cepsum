import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

# Operates 2 vocabularies for generation
# 1: Fixed target voabulary for P_gen
# 2: TODO: Attri vocabulary
class Myvocabulary:
    def __init__(self, vocab="vocab/vocab.txt"):
        # fixed vocabulary
        self.f_vocabs = []  
        self.f_wordToIndex = defaultdict(int)
        self.f_num = 0
        with open(vocab, "r", encoding="utf8") as f:
            for i, line in tqdm(enumerate(f)):
                values = line.split()
                self.vocabs.append(values[0])
                self.wordToIndex[values[0]] = i
                self.f_num += 1

    def getIndex(self, word):
        return self.f_wordToIndex[word]

    #not padded
    def getIndexes(self, lst, idx):
        #return [self.f_wordToIndex[i[idx]] for i in lst]
        return [self.f_wordToIndex[i[idx]] for i in lst]

    def getCandidateNum(self):
        return self.f_num