import torch
import json
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *
from tqdm import tqdm

def filtering(prob, k=50, p=0.8):
    # top-k filtering
    indices_to_remove = prob < torch.topk(prob, k)[0][..., -1, None]
    prob[indices_to_remove] = 0
    # top-p filtering
    sorted_prob, original_indices = torch.sort(prob, dim=-1, descending=True)
    remove_indices = torch.cumsum(sorted_prob, dim=-1) >= p
    remove_indices[0], remove_indices[1:] = False, remove_indices[:-1].clone()
    original_remove_indices = torch.zeros_like(prob, dtype=torch.bool) \
        .scatter_(dim=-1, index=original_indices, src=remove_indices)
    prob[original_remove_indices] = 0
    prob = prob / torch.sum(prob, dim=-1, keepdim=True)
    # return index
    return torch.multinomial(prob, 1).long()

def main():
    model = torch.load("")
    model.eval()

    embedding = MyEmbedding("cpu")

    with open("file_path", "r", encoding="utf8") as f:
        for i, line in tqdm(enumerate(f)):
            keys, values, src, tgt = line.split("\t\t")
            keys = [[key.split() for key in keys.split("\t")]]
            values = [[value.split() for value in values.split("\t")]]
            src, tgt = [src.split()], [""] * 128
            _, p_gen = model((keys, values, src, tgt))
            # Generate using p_gen
            gen_str = ""
            p_gen = p_gen.squeeze(1)    # (cur_length, candidate_size)
            for probs in p_gen:         # (candidate_size)
                index = filtering(probs)
                word = embedding.getWord(index)
                gen_str += word
            print(gen_str)
    
if __name__ == "__main__":
    main()
