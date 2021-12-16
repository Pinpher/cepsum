import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *

batch_size = 32
embed_dim = 300
hidden_size = 128
candidate_size = 1292610

#embedding = MyEmbedding()
#vocabulary = Myvocabulary()
data = MyDataset("data/cut_valid.txt")
dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)

def main():
    model = Mymodel(
        batch_size = batch_size,
        embed_dim = embed_dim,
        hidden_size = hidden_size,
        candidate_size = candidate_size
    )

    for i_batch, batch_data in enumerate(dataloader):
        loss = model(batch_data)
        break

    '''k_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    v_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    s_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    decoder = nn.LSTMCell(input_size=256, hidden_size=hidden_size)

    w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    v_a = nn.Linear(hidden_size, hidden_size, bias=False)
    u_a = nn.Linear(hidden_size, 1, bias=False)

    w_b = nn.Linear(hidden_size, candidate_size, bias=False)
    v_b = nn.Linear(2 * hidden_size, candidate_size, bias=False)    # Considering the wrong ctx shape

    for i_batch, batch_data in enumerate(dataloader):
        # batch_data[0]: (batch_size, num_keys,   num_words_in_key)
        # batch_data[1]: (batch_size, num_values, num_words_in_value)
        # batch_data[2]: (batch_size,             num_words_in_src)
        # batch_data[3]: (batch_size,             num_words_in_tgt)

        key_batch = [sum(i, []) for i in batch_data[0]]     # (batch_size, num_words)
        value_batch = [sum(i, []) for i in batch_data[1]]   # (batch_size, num_words)
        src_batch = batch_data[2]                           # (batch_size, num_words)
        tgt_batch = batch_data[3]                           # (batch_size, num_words)

        k_embedding, k_mask, _ = embedding.embed(key_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _
        v_embedding, v_mask, _ = embedding.embed(value_batch)  # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _
        s_embedding, s_mask, _ = embedding.embed(src_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1), _

        t_embedding, t_mask, t_indices = embedding.embed(tgt_batch)      # _ , (max_length, batch_size, 1), (max_length, batch_size)

        # Encode
        h_k, _ = k_encoder(k_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_v, _ = v_encoder(v_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_s, _ = s_encoder(s_embedding)     # (max_length, batch_size, hidden_size * 2)

        # Decode & Generate & Loss
        hidden_states = []
        loss = torch.zeros(batch_size)
        s = torch.zeros(batch_size, hidden_size) # s_0
        y = torch.zeros(batch_size, hidden_size) # y_0
        max_tgt_len = max(len(words) for words in tgt_batch)
        for t in range(1, max_tgt_len + 1):
            # Decede
            alpha = [u_a(torch.tanh(w_a(h_i) + v_a(s))) for h_i in h_s] # (max_length, batch_size, 1)
            alpha = F.softmax(torch.stack(alpha) * s_mask, dim=0)       # (max_length, batch_size, 1)
            ctx = torch.sum(alpha * h_s, dim=0)                         # (batch_size, hidden_size) (It is wrong!!!)
            s, y = decoder(ctx, (s, y))                                 # s_t and y_t (batch_size, hidden_size)
            hidden_states.append(s)                                     # (cur_length, batch_size, hidden_size)
            # Generate
            a = w_b(s)
            b = v_b(ctx)
            p_g = F.softmax(w_b(s) + v_b(ctx), dim=1)                   # P_gen (batch_size, candidate_size)
            # Loss
            index = t_indices[t - 1]                                    # (batch_size) list
            loss += -(torch.log(torch.stack([p_g[i][index[i]] for i in range(batch_size)]))) / max_tgt_len     # fking ugly (batch_size)

        #max_attr_len = max(len(keys) for keys in batch_data[0])
        #for t in range(1, max_attr_len + 1):
            # Decede

        # return loss
        break'''

if __name__ == "__main__":
    main()
    print("success")
