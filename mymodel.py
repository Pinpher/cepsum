# dual copy reserve

import re
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from myembedding import *

class Mymodel(nn.Module):
    def __init__(self, batch_size, embed_dim, hidden_size, device):
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

        self.w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(hidden_size, 1, bias=False)

        self.w_b = nn.Linear(hidden_size, self.candidate_size, bias=False)
        self.v_b = nn.Linear(hidden_size * 2, self.candidate_size, bias=False)

        '''self.w_d = nn.Linear(hidden_size * 2, 1, bias=False)
        self.u_d = nn.Linear(hidden_size, 1, bias=False)
        self.v_d = nn.Linear(hidden_size, 1, bias=False)

        self.w_e = nn.Linear(hidden_size * 2, 1, bias=False)
        self.u_e = nn.Linear(hidden_size, 1, bias=False)
        self.v_e = nn.Linear(hidden_size, 1, bias=False)'''

    def forward(self, batch_data):
        # batch_data[0]: (batch_size, num_keys,   num_words_in_key)
        # batch_data[1]: (batch_size, num_values, num_words_in_value)
        # batch_data[2]: (batch_size,             num_words_in_src)
        # batch_data[3]: (batch_size,             num_words_in_tgt)
        self.batch_size = len(batch_data[0])

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
        p, alpha_x, c_x, y_x, s_x = self.decode(decoder_s_h0, decoder_s_c0, t_embedding, s_mask, h_s, max_tgt_len)

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

        '''# turn each key into one (1, hidden_size * 2) tensor, like embedding
        # another ugly section
        h_k_attr = []
        for i in range(self.batch_size):
            h_k_attr_tmp = []
            cursor = 0
            for key in batch_data[0][i]:
                h_k_sum = 0
                for i in range(len(key)):
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

        _, alpha_k, c_k, _, _ = self.decode(mask_key_attr, h_k_attr, max_tgt_len)
        _, alpha_v, c_v, _, _ = self.decode(v_mask, h_v, max_tgt_len)

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

        gama_t = torch.sigmoid(self.w_d(c_k) + self.w_d(c_x) + self.u_d(s_x) + self.v_d(y_x))           # (max_tgt_len, batch_size, 1)
        p_copy = gama_t * p_copy_x + (torch.ones(gama_t.shape).to(self.device) - gama_t) * p_copy_v     # (max_tgt_len, batch_size, candidate_size)

        lambda_t = torch.sigmoid(self.w_e(c_x) + self.u_e(s_x) + self.v_e(y_x))                         # (max_tgt_len, batch_size, 1)
        p = lambda_t * p_gen +  (torch.ones(lambda_t.shape).to(self.device) - lambda_t) * p_copy        # (max_tgt_len, batch_size, candidate_size)'''
        
        return loss, p

    # Decode & Generate
    def decode(self, hidden, cell, t_embedding, input_mask, input_h, max_input_len):
        p, alpha_per_t, c_per_t, y_per_t, s_per_t = [], [], [], [], []
        # For each target word
        for t in range(max_input_len):
            # y_per_t.append(y)                                           # y_per_t indicates y_(t-1) (cur_length, batch_size, hidden_size)
            
            # Decode
            alpha = [self.u_a(torch.tanh(self.w_a(h_i) + self.v_a(hidden))) for h_i in input_h] # (max_length, batch_size, 1)
            alpha = F.softmax(torch.stack(alpha).to(self.device) * input_mask, dim=0)           # (max_length, batch_size, 1)
            ct = torch.sum(alpha * input_h, dim=0)                                              # (batch_size, hidden_size * 2)
            hidden, cell = self.decoder(t_embedding[t], (hidden, cell))                         # (batch_size, hidden_size)
            #s_per_t.append(s)                                           # (cur_length, batch_size, hidden_size)
            #alpha_per_t.append(alpha)                                   # (cur_length, max_length, batch_size, 1)
            #c_per_t.append(ct)                                          # (cur_length, batch_size, hidden_size * 2)
            # Generate
            p.append(F.softmax(self.w_b(hidden) + self.v_b(ct), dim=1))                         # only for p_gen & src (cur_length, batch_size, candidate_size)
        return torch.stack(p), torch.stack(alpha_per_t), torch.stack(c_per_t), torch.stack(y_per_t), torch.stack(s_per_t)