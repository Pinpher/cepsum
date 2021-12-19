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
    return probs, hidden, cell, alpha, ct

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
    model.load_state_dict(torch.load("./model/model_copy_7"))
    model.embedding.embedding.load_state_dict(torch.load("./model/model_copy_7_embedding"))
    model.to("cuda")
    model.eval()

    # f is a "cut_" file
    fout = open("./data/gen_copy_valid.txt", "w", encoding="utf8")

    with open("./data/cut_valid.txt", "r", encoding="utf8") as f:
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

            # Decoder key & value init state
            cell_k = torch.cat(list(state_k[1]), dim=1)             # (batch_size, hidden_size * 2)
            decoder_k_c0 = model.decoder_k_init(cell_k)             # (batch_size, hidden_size)
            decoder_k_h0 = torch.tanh(decoder_k_c0)                 # (batch_size, hidden_size)
            cell_v = torch.cat(list(state_v[1]), dim=1)             # (batch_size, hidden_size * 2)
            decoder_v_c0 = model.decoder_v_init(cell_v)             # (batch_size, hidden_size)
            decoder_v_h0 = torch.tanh(decoder_v_c0)                 # (batch_size, hidden_size)

            # turn each key into one (1, hidden_size * 2) tensor, like embedding
            # another ugly section
            h_k_attr = []
            h_k_attr_tmp = []
            cursor = 0
            for key in batch_data[0][0]:
                h_k_sum = 0
                for j in range(len(key)):
                    h_k_sum += h_k[cursor, 0]
                    cursor += 1
                h_k_attr_tmp.append(h_k_sum)
            h_k_attr.append(h_k_attr_tmp)
            # h_k_attr now is (batch_size, num_keys, hidden_size * 2)

            # pad & mask for key
            max_key_len = max(len(keys) for keys in h_k_attr)
            mask_key_attr = []
            h_k_attr[0] += [torch.zeros(model.hidden_size * 2).to(model.device)] * (max_key_len - len(h_k_attr[0]))
            mask_key_attr.append(torch.IntTensor([1] * len(h_k_attr[0]) + [0] * (max_key_len - len(h_k_attr[0]))).to(model.device))
            h_k_attr = torch.stack([torch.stack(i) for i in h_k_attr]).transpose(0, 1)  # (max_key_length, batch_size, hidden_size * 2)
            mask_key_attr = torch.stack(mask_key_attr).transpose(0, 1).unsqueeze(-1)    # (max_key_length, batch_size, 1)

            # Generate
            gen_str = ""
            last_word = "[CLS]"
            max_tgt_len = 128
            hidden_s, cell_s = decoder_s_h0, decoder_s_c0 
            hidden_k, cell_k = decoder_k_h0, decoder_k_c0 
            hidden_v, cell_v = decoder_v_h0, decoder_v_c0 
            for i in range(max_tgt_len):
                # p_gen
                probs_gen, hidden_s, cell_s, alpha_x, c_x = decode_step(model, hidden_s, cell_s, last_word, s_mask, h_s)
                
                # p_copy_x
                probs_copy_x = torch.zeros(1, model.candidate_size).to(model.device)
                for i, index in enumerate(s_indices):
                    mask = s_mask[i].squeeze(-1)    # (batch_size)
                    probs_copy_x[range(1), index.long()] += (alpha_x[i].squeeze(-1) * mask)    # (batch_size, candidate_size)

                # p_copy_v
                _, hidden_k, cell_k, alpha_k, c_k = decode_step(model, hidden_k, cell_k, last_word, mask_key_attr, h_k_attr)
                _, hidden_v, cell_v, alpha_v, c_v = decode_step(model, hidden_v, cell_v, last_word, v_mask, h_v)
                probs_copy_v = torch.zeros(1, model.candidate_size).to(model.device)    # (batch_size, candidate_size)
                attr_i, attr_j = 0, 0
                attr_sizes = [len(value) for value in batch_data[1][0]]                 # (num_keys)
                for i, index in enumerate(v_indices[:,0]):
                    if index == model.embedding.sepId:
                        break
                    if attr_j == attr_sizes[attr_i]:
                        attr_i += 1
                        attr_j = 0                    
                    mask_k = k_mask[attr_i][0].squeeze(-1)    # (1)
                    mask_v = v_mask[i][0].squeeze(-1)         # (1)
                    probs_copy_v[0, index.long()] += (alpha_k[attr_i][0].squeeze(-1) * mask_k) * (alpha_v[i][0].squeeze(-1) * mask_v)    # (batch_size, candidate_size)
                    attr_j += 1

                gamma_t = torch.sigmoid(model.w_d(c_k) + model.w_d(c_x) + model.u_d(hidden_s) + model.v_d(model.embedding.embed([[last_word]])[0][0]))          # (batch_size, 1)
                probs_copy = gamma_t * probs_copy_x + (torch.ones(gamma_t.shape).to(model.device) - gamma_t) * probs_copy_v                                     # (batch_size, candidate_size)

                lambda_t = torch.sigmoid(model.w_e(c_x) + model.u_e(hidden_s) + model.v_e(model.embedding.embed([[last_word]])[0][0]))                          # (batch_size, 1)
                probs = lambda_t * probs_gen +  (torch.ones(lambda_t.shape).to(model.device) - lambda_t) * probs_copy                                           # (batch_size, candidate_size)

                # final probs
                index = filtering(probs.squeeze(0).detach())
                last_word = model.embedding.getWord(index)
                if last_word  == "[SEP]":
                    break
                if last_word  == "[UNK]":
                    continue
                gen_str += last_word
            #print(gen_str)
            if flag:
                #print(gen_str)
                fout.write(gen_str + "\n")
            
    fout.close()
    
if __name__ == "__main__":
    main()
