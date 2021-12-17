import torch
import torch.nn as nn
import torch.nn.functional as F
from myembedding import *

class Mymodel(nn.Module):
    def __init__(self, batch_size, embed_dim, hidden_size, candidate_size, device):
        super(Mymodel, self).__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.candidate_size = candidate_size
        self.device = device

        self.embedding = MyEmbedding()

        self.k_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.v_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.s_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=256, hidden_size=hidden_size)

        self.w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(hidden_size, 1, bias=False)

        self.w_b = nn.Linear(hidden_size, candidate_size, bias=False)
        self.v_b = nn.Linear(hidden_size * 2, candidate_size, bias=False)   # Considering the wrong ctx shape


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
        s_mask = s_mask.to(self.device)

        t_embedding, t_mask, t_indices = self.embedding.embed(tgt_batch)  
        t_mask = t_mask.to(self.device)

        # Encode
        h_k, _ = self.k_encoder(k_embedding.to(self.device))     # (max_length, batch_size, hidden_size * 2)
        h_v, _ = self.v_encoder(v_embedding.to(self.device))     # (max_length, batch_size, hidden_size * 2)
        h_s, _ = self.s_encoder(s_embedding.to(self.device))     # (max_length, batch_size, hidden_size * 2)

        # Decode & Generate & Loss
        hidden_states = []
        loss = []
        s = torch.zeros(self.batch_size, self.hidden_size).to(self.device) # s_0
        y = torch.zeros(self.batch_size, self.hidden_size).to(self.device) # y_0
        max_tgt_len = max(len(words) for words in tgt_batch)
        for t in range(1, max_tgt_len + 1):
            # Decede
            alpha = [self.u_a(torch.tanh(self.w_a(h_i) + self.v_a(s))) for h_i in h_s] # (max_length, batch_size, 1)
            alpha = F.softmax(torch.stack(alpha) * s_mask, dim=0)       # (max_length, batch_size, 1)
            ctx = torch.sum(alpha * h_s, dim=0)                         # (batch_size, hidden_size) (It is wrong!!!)
            s, y = self.decoder(ctx, (s, y))                            # s_t and y_t (batch_size, hidden_size)
            hidden_states.append(s)                                     # (cur_length, batch_size, hidden_size)
            # Generate
            p_g = F.softmax(self.w_b(s) + self.v_b(ctx), dim=1)         # P_gen (batch_size, candidate_size)
        
        '''for t in range(1, max_tgt_len + 1):
            # Loss
            index = t_indices[t - 1]                                    # (batch_size) list
            loss += -(torch.log(torch.stack([p_g[i][index[i]] for i in range(self.batch_size)]))) / max_tgt_len     # fking ugly (batch_size)
            tmp_loss = -(torch.log(torch.stack([p_g[i][index[i]] for i in range(self.batch_size)])))'''

        for i in range(self.batch_size):
            indices = t_indices[:,i]
            mask = t_mask[:,i].squeeze(-1).byte()                        
            indices = torch.masked_select(indices, mask)                
            loss.append(torch.mean(-torch.log(torch.Tensor([p_g[i][index] for index in indices]))))

        return loss.mean()
