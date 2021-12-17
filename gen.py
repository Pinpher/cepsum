import torch
import json
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *
from tqdm import tqdm

def main():
    model = torch.load("")
    model.eval()

    embedding = MyEmbedding("cpu")

    with open("file_path", "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            keys, values, src, tgt = line.split("\t\t")
            keys = [[key.split() for key in keys.split("\t")]]
            values = [[value.split() for value in values.split("\t")]]
            src, tgt = [src.split()], [tgt.split()]

            # TODO
    
if __name__ == "__main__":
    main()
