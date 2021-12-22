import enum
import torch
import json
import argparse
import random
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--module_dict', type=str, default="./model/model_only_copy_final_1")
parser.add_argument('--input_path', type=str, default="./data/cut_valid.txt")
parser.add_argument('--output_path', type=str, default="./data/gen_only_copy_final_valid.txt")
parser.add_argument('--attri_words_path', type=str, default='./vocab/simple_attr_words.txt')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--min_length', type=int, default=48)
args = parser.parse_args()


def decode_step(model, hidden, cell, last_word, input_mask, input_h, candidate_mask):
    alpha = [model.u_a(torch.tanh(model.w_a(h_i) + model.v_a(hidden))) for h_i in input_h]          # (max_length, batch_size, 1)
    alpha = F.softmax(torch.stack(alpha).to(model.device) * input_mask, dim=0)                      # (max_length, batch_size, 1)
    ct = torch.sum(alpha * input_h, dim=0)                                                          # (batch_size, hidden_size * 2)
    hidden, cell = model.decoder(model.embedding_tgt.embed([[last_word]])[0][0], (hidden, cell))    # (batch_size, hidden_size)
    
    if candidate_mask != None:
        logits = model.u_b(model.dropout(torch.tanh(model.w_b(hidden) + model.v_b(ct))))
        logits = torch.masked_fill(logits, candidate_mask.bool(), -float('inf'))
        probs = F.softmax(logits, dim=1)
        return probs, hidden, cell, alpha, ct
    else:
        return 0, hidden, cell, alpha, ct

def filtering(prob, k=25, p=0.8):
    # top-k filtering
    if k > 0:
        indices_to_remove = prob < torch.topk(prob, k)[0][..., -1, None]
        prob[indices_to_remove] = 0
        prob = prob / torch.sum(prob, dim=-1, keepdim=True)
    # top-p filtering
    sorted_prob, original_indices = torch.sort(prob, dim=-1, descending=True)
    remove_indices = torch.cumsum(sorted_prob, dim=-1) >= p
    remove_indices[0], remove_indices[1:] = False, remove_indices[:-1].clone()
    original_remove_indices = torch.zeros_like(prob, dtype=torch.bool) \
        .scatter_(dim=-1, index=original_indices, src=remove_indices)
    prob[original_remove_indices] = 0
    prob = prob / torch.sum(prob, dim=-1, keepdim=True)
    # return prob distribution
    return prob

def main():
    model = Mymodel(
        batch_size = 32,
        embed_dim = 300,
        hidden_size = args.hidden_size,
        device = "cuda",
        attri_words_path = args.attri_words_path
    )
    model.load_state_dict(torch.load(args.module_dict))
    model.embedding.embedding.load_state_dict(torch.load(args.module_dict + "_embedding"))
    model.embedding_tgt.embedding.load_state_dict(torch.load(args.module_dict + "_embedding"))
    model.to("cuda")
    model.eval()

    # f is a "cut_" file
    fout = open(args.output_path, "w", encoding="utf8")

    with open(args.input_path, "r", encoding="utf8") as f:
        last = ""
        flag = False
        for i, line in tqdm(enumerate(f)):
            keys, values, src, tgt = line.split("\t\t")
            flag = (last != src)
            last = src

            if not flag:
                continue
            
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
            last_probs = [1]
            last_words = ["[CLS]"]
            last_gen_str = [""]
            finished = [False]
            finished_cnt = 0
            max_tgt_len = 128
            hidden_s, cell_s = decoder_s_h0, decoder_s_c0 
            hidden_k, cell_k = decoder_k_h0, decoder_k_c0 
            hidden_v, cell_v = decoder_v_h0, decoder_v_c0 
            for _ in range(max_tgt_len):
                cur_probs = [0] * len(last_words)
                for idx, last_word in enumerate(last_words):
                    if finished[idx]:
                        continue

                    # p_gen
                    probs_gen, hidden_s, cell_s, alpha_x, c_x = decode_step(model, hidden_s, cell_s, last_word, s_mask, h_s, model.candidate_mask.to(model.device))
                    
                    # p_copy_x
                    probs_copy_x = torch.zeros(1, model.candidate_size).to(model.device)
                    for i, index in enumerate(s_indices):
                        mask = s_mask[i].squeeze(-1)    # (batch_size)
                        probs_copy_x[range(1), index.long()] += (alpha_x[i].squeeze(-1) * mask)    # (batch_size, candidate_size)

                    # p_copy_v
                    _, hidden_k, cell_k, alpha_k, c_k = decode_step(model, hidden_k, cell_k, last_word, mask_key_attr, h_k_attr, None)
                    _, hidden_v, cell_v, alpha_v, c_v = decode_step(model, hidden_v, cell_v, last_word, v_mask, h_v, None)
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

                    gamma_t = torch.sigmoid(model.w_d(c_k) + model.w_d(c_x) + model.u_d(hidden_s) + model.v_d(model.embedding_tgt.embed([[last_word]])[0][0]))          # (batch_size, 1)
                    probs_copy = gamma_t * probs_copy_x + (torch.ones(gamma_t.shape).to(model.device) - gamma_t) * probs_copy_v                                     # (batch_size, candidate_size)

                    lambda_t = torch.sigmoid(model.w_e(c_x) + model.u_e(hidden_s) + model.v_e(model.embedding_tgt.embed([[last_word]])[0][0]))                          # (batch_size, 1)
                    probs = lambda_t * probs_gen + (torch.ones(lambda_t.shape).to(model.device) - lambda_t) * probs_copy                                           # (batch_size, candidate_size)

                    cur_probs[idx] = probs
                
                for idx, last_prob in enumerate(last_probs):
                    if finished[idx]:
                        continue
                    #cur_probs[idx] = cur_probs[idx] * last_prob
                    cur_probs[idx] = cur_probs[idx].reshape(-1) # (candidate_size)
                
                flatten_probs = torch.cat(cur_probs[finished_cnt:])        # unfinished * candidate_size
                _, indices = torch.topk(flatten_probs, 2 * (args.beam_size - finished_cnt))
                new_probs = []
                new_words = []
                new_gen_str = []
                new_finished = []
                cur_add_num = 0
                for index in indices:
                    x = index // model.candidate_size + finished_cnt
                    y = index % model.candidate_size
                    if model.embedding_tgt.getWord(y) == "[SEP]" and len(last_gen_str[x]) < args.min_length:
                        continue
                    cur_add_num += 1
                    new_probs.append(cur_probs[x][y].cpu().item())
                    new_words.append(model.embedding_tgt.getWord(y))
                    new_gen_str.append(last_gen_str[x] + new_words[-1])
                    new_finished.append(new_words[-1] == "[SEP]")
                    if cur_add_num == args.beam_size - finished_cnt:
                        break
                # keep the finished part
                last_probs = last_probs[:finished_cnt]
                last_words = last_words[:finished_cnt]
                last_gen_str = last_gen_str[:finished_cnt]
                finished = finished[:finished_cnt]
                # add new part
                for idx, finish_flag in enumerate(new_finished):
                    if finish_flag:
                        last_probs.append(new_probs[idx])
                        last_words.append(new_words[idx])
                        last_gen_str.append(new_gen_str[idx])
                        finished.append(finish_flag)
                for idx, finish_flag in enumerate(new_finished):
                    if not finish_flag:
                        last_probs.append(new_probs[idx])
                        last_words.append(new_words[idx])
                        last_gen_str.append(new_gen_str[idx])
                        finished.append(finish_flag)
                finished_cnt = sum(finished)
                if all(finished):
                    break
            print(last_gen_str)
            #fout.write(random.choice(last_gen_str).strip("[SEP]") + "\n")
            
    fout.close()
    
if __name__ == "__main__":
    main()
