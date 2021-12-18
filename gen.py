import torch
import json
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *
from tqdm import tqdm

def decode_step(model, hidden, cell, last_word, input_mask, input_h):
    alpha = [model.u_a(torch.tanh(model.w_a(h_i) + model.v_a(hidden))) for h_i in input_h]  # (max_length, batch_size, 1)
    alpha = F.softmax(torch.stack(alpha).to(model.device) * input_mask, dim=0)              # (max_length, batch_size, 1)
    ct = torch.sum(alpha * input_h, dim=0)                                                  # (batch_size, hidden_size * 2)
    hidden, cell = model.decoder(model.embedding.embed([[last_word]])[0][0], (hidden, cell))   # (batch_size, hidden_size)
    probs = F.softmax(model.w_b(hidden) + model.v_b(ct), dim=1)                             # This is only p_gen 
    return probs, hidden, cell

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
    model = Mymodel(
        batch_size = 32,
        embed_dim = 300,
        hidden_size = 512,
        device = "cuda"
    )
    model.load_state_dict(torch.load("./model/model_copy_3"))
    model.embedding.embedding.load_state_dict(torch.load("./model/model_copy_3_embedding"))
    model.to("cuda")
    model.eval()

    # f is a "cut_" file
    fout = open("./data/generate_tgt.txt", "w", encoding="utf8")

    with open("./data/cut_test.txt", "r", encoding="utf8") as f:
        last = ""
        flag = False
        for i, line in tqdm(enumerate(f)):
            keys, values, src, tgt = line.split("\t\t")
            flag = (last != src)
            last = src
            keys = [[key.split() for key in keys.split("\t")]]
            values = [[value.split() for value in values.split("\t")]]
            src = [src.split()]

            batch_data = keys, values, src
            # Generate
            key_batch = [sum(i, []) for i in batch_data[0]]     # (batch_size, num_words)
            value_batch = [sum(i, []) for i in batch_data[1]]   # (batch_size, num_words)
            src_batch = batch_data[2]                           # (batch_size, num_words)

            k_embedding, k_mask, k_indices = model.embedding.embed(key_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)
            v_embedding, v_mask, v_indices = model.embedding.embed(value_batch)  # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)
            s_embedding, s_mask, s_indices = model.embedding.embed(src_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)

            # Encode
            h_k, state_k = model.k_encoder(k_embedding)              # (max_length, batch_size, hidden_size * 2)
            h_v, state_v = model.v_encoder(v_embedding)              # (max_length, batch_size, hidden_size * 2)
            h_s, state_s = model.s_encoder(s_embedding)              # (max_length, batch_size, hidden_size * 2)

            # Decoder init state
            cell_s = torch.cat(list(state_s[1]), dim=1)             # (batch_size, hidden_size * 2)
            decoder_s_c0 = model.decoder_s_init(cell_s)             # (batch_size, hidden_size)
            decoder_s_h0 = torch.tanh(decoder_s_c0)                 # (batch_size, hidden_size)

            # Generate
            gen_str = ""
            last_word = "[CLS]"
            max_tgt_len = 128
            h, c = decoder_s_h0, decoder_s_c0 
            for i in range(max_tgt_len):
                probs, h, c = decode_step(model, h, c, last_word, s_mask, h_s)
                index = filtering(probs.squeeze(0).detach())
                last_word = model.embedding.getWord(index)
                if last_word  == "[SEP]":
                    break
                gen_str += last_word
            #print(gen_str)
            if flag:
                fout.write(gen_str + "\n")
            

    fout.close()
    
if __name__ == "__main__":
    main()
