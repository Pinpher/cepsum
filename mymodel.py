# dual copy reserve

import re
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from myembedding import *

class Mymodel(nn.Module):
    def __init__(self, batch_size, embed_dim, hidden_size, device, attri_words_path):
        super(Mymodel, self).__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = MyEmbedding(self.device)

        self.candidate_size = self.embedding.vocabSize()

        self.k_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.v_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.s_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)

        self.decoder_s_init = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder_k_init = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder_v_init = nn.Linear(hidden_size * 2, hidden_size)

        self.w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(hidden_size, 1, bias=False)

        self.w_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_b = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.u_b = nn.Linear(hidden_size, self.candidate_size, bias=False)

        self.w_d = nn.Linear(hidden_size * 2, 1, bias=False)
        self.u_d = nn.Linear(hidden_size, 1, bias=False)
        self.v_d = nn.Linear(embed_dim, 1, bias=False)

        self.w_e = nn.Linear(hidden_size * 2, 1, bias=False)
        self.u_e = nn.Linear(hidden_size, 1, bias=False)
        self.v_e = nn.Linear(embed_dim, 1, bias=False)

        self.v_e = nn.Linear(embed_dim, 1, bias=False)

        attr_file = open(attri_words_path, "r", encoding='utf-8')
        attr_words = attr_file.readlines()                              # (attr_word_num)
        attr_file.close()
        _, _, attr_words_indices = self.embedding.embed([attr_words])   # (attr_word_num, 1)
        attr_words_indices = attr_words_indices.squeeze(-1)
        self.candidate_mask = torch.zeros(self.candidate_size)
        self.candidate_mask[attr_words_indices.long()] = 1              # (candidate_size)

    def forward(self, batch_data):
        # batch_data[0]: (batch_size, num_keys,   num_words_in_key)
        # batch_data[1]: (batch_size, num_values, num_words_in_value)
        # batch_data[2]: (batch_size,             num_words_in_src)
        # batch_data[3]: (batch_size,             num_words_in_tgt)
        self.batch_size = len(batch_data[0])
        batch_candidate_mask = torch.stack([self.candidate_mask] * self.batch_size, dim=0).to(self.device)

        key_batch = [sum(i, []) for i in batch_data[0]]     # (batch_size, num_words)
        value_batch = [sum(i, []) for i in batch_data[1]]   # (batch_size, num_words)
        src_batch = batch_data[2]                           # (batch_size, num_words)
        tgt_batch = batch_data[3]                           # (batch_size, num_words)

        k_embedding, k_mask, k_indices = self.embedding.embed(key_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)
        v_embedding, v_mask, v_indices = self.embedding.embed(value_batch)  # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)
        s_embedding, s_mask, s_indices = self.embedding.embed(src_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)
        t_embedding, t_mask, t_indices = self.embedding.embed(tgt_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), (max_length, batch_size)

        # Encode
        h_k, state_k = self.k_encoder(k_embedding)              # (max_length, batch_size, hidden_size * 2)
        h_v, state_v = self.v_encoder(v_embedding)              # (max_length, batch_size, hidden_size * 2)
        h_s, state_s = self.s_encoder(s_embedding)              # (max_length, batch_size, hidden_size * 2)

        # Decoder init state
        cell_s = torch.cat(list(state_s[1]), dim=1)             # (batch_size, hidden_size * 2)
        decoder_s_c0 = self.decoder_s_init(cell_s)              # (batch_size, hidden_size)
        decoder_s_h0 = torch.tanh(decoder_s_c0)                 # (batch_size, hidden_size)

        # Decode
        max_tgt_len = max(len(words) for words in tgt_batch)
        max_v_len = max(len(words) for words in value_batch)
        p_gen, alpha_x, c_x, s_x = self.decode(decoder_s_h0, decoder_s_c0, t_embedding, s_mask, h_s, max_tgt_len, batch_candidate_mask)

        # turn each key into one (1, hidden_size * 2) tensor, like embedding
        # another ugly section
        h_k_attr = []
        for i in range(self.batch_size):
            h_k_attr_tmp = []
            cursor = 0
            for key in batch_data[0][i]:
                h_k_sum = 0
                for j in range(len(key)):
                    h_k_sum += h_k[cursor, i]
                    cursor += 1
                h_k_attr_tmp.append(h_k_sum)
            h_k_attr.append(h_k_attr_tmp)
        # h_k_attr now is (batch_size, num_keys, hidden_size * 2)

        # pad & mask for key
        max_key_len = max(len(keys) for keys in h_k_attr)
        mask_key_attr = []
        for i in range(self.batch_size):
            h_k_attr[i] += [torch.zeros(self.hidden_size * 2).to(self.device)] * (max_key_len - len(h_k_attr[i]))
            mask_key_attr.append(torch.IntTensor([1] * len(h_k_attr[i]) + [0] * (max_key_len - len(h_k_attr[i]))).to(self.device))
        h_k_attr = torch.stack([torch.stack(i) for i in h_k_attr]).transpose(0, 1)  # (max_key_length, batch_size, hidden_size * 2)
        mask_key_attr = torch.stack(mask_key_attr).transpose(0, 1).unsqueeze(-1)    # (max_key_length, batch_size, 1)

        # Decoder key & value init state
        cell_k = torch.cat(list(state_k[1]), dim=1)             # (batch_size, hidden_size * 2)
        decoder_k_c0 = self.decoder_k_init(cell_k)              # (batch_size, hidden_size)
        decoder_k_h0 = torch.tanh(decoder_k_c0)                 # (batch_size, hidden_size)
        cell_v = torch.cat(list(state_v[1]), dim=1)             # (batch_size, hidden_size * 2)
        decoder_v_c0 = self.decoder_v_init(cell_v)              # (batch_size, hidden_size)
        decoder_v_h0 = torch.tanh(decoder_v_c0)                 # (batch_size, hidden_size)

        _, alpha_k, c_k, s_k = self.decode(decoder_k_h0, decoder_k_c0, t_embedding, mask_key_attr, h_k_attr, max_tgt_len, None)
        _, alpha_v, c_v, s_v = self.decode(decoder_v_h0, decoder_v_c0, t_embedding, v_mask, h_v, max_tgt_len, None)

        p_copy_x = []
        for t in range(0, max_tgt_len):
            alpha = alpha_x[t]      # (max_src_length, batch_size, 1)
            P_copy_x_per_t = torch.zeros(self.batch_size, self.candidate_size).to(self.device)  # (batch_size, candidate_size)
            for i, index in enumerate(s_indices):
                mask = s_mask[i].squeeze(-1)    # (batch_size)
                P_copy_x_per_t[range(self.batch_size), index.long()] += (alpha[i].squeeze(-1) * mask)    # (batch_size, candidate_size)
            p_copy_x.append(P_copy_x_per_t)
        p_copy_x = torch.stack(p_copy_x)    # (max_tgt_len, batch_size, candidate_size)

        p_copy_v = []
        for t in range(0, max_tgt_len):
            alpha_k_per_t = alpha_k[t]      # (max_attri_length, batch_size, 1)
            alpha_v_per_t = alpha_v[t]      # (max_value_length, batch_size, 1)
            p_copy_v_per_t = torch.zeros(self.batch_size, self.candidate_size).to(self.device)  # (batch_size, candidate_size)
            for batch_i in range(self.batch_size):
                attr_i, attr_j = 0, 0
                attr_sizes = [len(value) for value in batch_data[1][batch_i]]   # (num_keys)
                for i, index in enumerate(v_indices[:,batch_i]):
                    if index == self.embedding.sepId:
                        break
                    if attr_j == attr_sizes[attr_i]:
                        attr_i += 1
                        attr_j = 0                    
                    mask_k = k_mask[attr_i][batch_i].squeeze(-1)    # (1)
                    mask_v = v_mask[i][batch_i].squeeze(-1)         # (1)
                    p_copy_v_per_t[batch_i, index.long()] += (alpha_k_per_t[attr_i][batch_i].squeeze(-1) * mask_k) * (alpha_v_per_t[i][batch_i].squeeze(-1) * mask_v)    # (batch_size, candidate_size)
                    attr_j += 1
            p_copy_v.append(p_copy_v_per_t)
        p_copy_v = torch.stack(p_copy_v)    # (max_tgt_len, batch_size, candidate_size)

        gamma_t = torch.sigmoid(self.w_d(c_k) + self.w_d(c_x) + self.u_d(s_x) + self.v_d(t_embedding))  # (max_tgt_len, batch_size, 1)
        p_copy = gamma_t * p_copy_x + (torch.ones(gamma_t.shape).to(self.device) - gamma_t) * p_copy_v  # (max_tgt_len, batch_size, candidate_size)

        lambda_t = torch.sigmoid(self.w_e(c_x) + self.u_e(s_x) + self.v_e(t_embedding))                 # (max_tgt_len, batch_size, 1)
        p = lambda_t * p_gen +  (torch.ones(lambda_t.shape).to(self.device) - lambda_t) * p_copy        # (max_tgt_len, batch_size, candidate_size)

        # loss
        loss = torch.zeros(self.batch_size).to(self.device)
        for i in range(self.batch_size):
            p_i = p[:, i].squeeze(1)                            # (max_length, candidate_size)
            mask_i = t_mask[:, i]                               # (max_length, 1)
            p_i = torch.masked_select(p_i, mask_i)              # (length_i * candidate_size)
            p_i = p_i.reshape(-1, self.candidate_size)          # (length_i, candidate_size)
            # The following line is needed
            mask_i = torch.cat((mask_i[-1:], mask_i[:-1]))      # (max_length, 1)
            y_i = t_indices[:, i].unsqueeze(-1)                 # (max_length, 1)
            y_i = torch.masked_select(y_i, mask_i)              # (length_i)
            probs = p_i[range(len(y_i)), y_i.long()]            # (length_i)
            loss[i] = torch.mean(-torch.log(probs))

        return loss, p

    # Attention & Decode & Generate
    def decode(self, hidden, cell, t_embedding, input_mask, input_h, max_input_len, batch_candidate_mask):
        p, alpha, c, s = [], [], [], []
        # For each target word
        for t in range(max_input_len):
            # Decode
            alpha_t = [self.u_a(torch.tanh(self.w_a(h_i) + self.v_a(hidden))) for h_i in input_h]       # (max_length, batch_size, 1)
            alpha_t = F.softmax(torch.stack(alpha_t).to(self.device) * input_mask, dim=0)               # (max_length, batch_size, 1)
            c_t = torch.sum(alpha_t * input_h, dim=0)                                                   # (batch_size, hidden_size * 2)
            hidden, cell = self.decoder(t_embedding[t], (hidden, cell))                                 # (batch_size, hidden_size)
            # Generate
            alpha.append(alpha_t)
            c.append(c_t)
            s.append(hidden)

        if batch_candidate_mask != None:
            for t in range(max_input_len):
                logits = self.u_b(torch.tanh(self.w_b(s[t]) + self.v_b(c[t])))
                logits = torch.masked_fill(logits, batch_candidate_mask, -float('inf'))
                # only for p_gen & src (cur_length, batch_size, candidate_size)
                p.append(F.softmax(logits, dim=1))
            return torch.stack(p), torch.stack(alpha), torch.stack(c), torch.stack(s)
        else:
            return 0, torch.stack(alpha), torch.stack(c), torch.stack(s)