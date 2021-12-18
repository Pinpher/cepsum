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
        self.decoder = nn.LSTMCell(input_size=hidden_size * 2, hidden_size=hidden_size)

        self.w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(hidden_size, 1, bias=False)

        self.w_b = nn.Linear(hidden_size, self.candidate_size, bias=False)
        self.v_b = nn.Linear(hidden_size * 2, self.candidate_size, bias=False)


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

        k_embedding, k_mask, _ = self.embedding.embed(key_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _
        v_embedding, v_mask, _ = self.embedding.embed(value_batch)  # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _
        s_embedding, s_mask, _ = self.embedding.embed(src_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _
        t_embedding, t_mask, t_indices = self.embedding.embed(tgt_batch)  

        # Encode
        h_k, _ = self.k_encoder(k_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_v, _ = self.v_encoder(v_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_s, _ = self.s_encoder(s_embedding)     # (max_length, batch_size, hidden_size * 2)

        max_tgt_len = max(len(words) for words in tgt_batch)
        p_gen, alpha_x, _ = self.decode(s_mask, h_s, max_tgt_len)

        # loss
        loss = torch.zeros(self.batch_size).to(self.device)
        p_gen = torch.stack(p_gen)
        for i in range(self.batch_size):
            p_i = p_gen[:, i].squeeze(1)                        # (max_length, candidate_size)
            mask_i = t_mask[:, i]                               # (max_length, 1)
            p_i = torch.masked_select(p_i, mask_i)              # (length_i * candidate_size)
            p_i = p_i.reshape(-1, self.candidate_size)          # (length_i, candidate_size)
            y_i = t_indices[:, i].unsqueeze(-1)                 # (max_length)
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
                tmp1 = 0
                for i in range(len(key)):
                    tmp1 += h_k[cursor, i]
                    cursor += 1
                h_k_attr_tmp.append(tmp1)
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

        _, alpha_k, c_k = self.decode(mask_key_attr, h_k_attr, max_key_len)'''

        return loss, p_gen

    # Decode & Generate
    def decode(self, input_mask, input_h, max_input_len):
        p, alpha_per_t, c_per_t = [], [], []
        s = torch.zeros(self.batch_size, self.hidden_size).to(self.device) # s_0
        y = torch.zeros(self.batch_size, self.hidden_size).to(self.device) # y_0
        for t in range(1, max_input_len + 1):
            # Decede
            alpha = [self.u_a(torch.tanh(self.w_a(h_i) + self.v_a(s))) for h_i in input_h]      # (max_length, batch_size, 1)
            alpha = F.softmax(torch.stack(alpha).to(self.device) * input_mask, dim=0)           # (max_length, batch_size, 1)
            ct = torch.sum(alpha * input_h, dim=0)                      # (batch_size, hidden_size * 2)
            s, y = self.decoder(ct, (s, y))                             # s_t and y_t (batch_size, hidden_size)
            alpha_per_t.append(alpha)                                   # (cur_length, max_length, batch_size, 1)
            c_per_t.append(ct)                                          # (cur_length, hidden_size * 2)
            # Generate
            p.append(F.softmax(self.w_b(s) + self.v_b(ct), dim=1))      # only for p_gen & src (cur_length, batch_size, candidate_size)
        return p, alpha_per_t, c_per_t