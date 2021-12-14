import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from myembedding import *
from mydataset import *

batch_size = 32
embed_dim = 300
hidden_size = 128

embedding = MyEmbedding()
data = MyDataset("data/cut_valid.txt")
dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)

def main():
    k_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    v_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    s_encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
    decoder = nn.LSTMCell(input_size=256, hidden_size=hidden_size)

    w_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    v_a = nn.Linear(hidden_size, hidden_size, bias=False)
    u_a = nn.Linear(hidden_size, 1, bias=False)

    for i_batch, batch_data in enumerate(dataloader):
        # batch_data[0]: (batch_size, num_keys,   num_words_in_key)
        # batch_data[1]: (batch_size, num_values, num_words_in_value)
        # batch_data[2]: (batch_size,             num_words_in_src)
        # batch_data[3]: (batch_size,             num_words_in_tgt)

        key_batch = [sum(i, []) for i in batch_data[0]]     # (batch_size, num_words)
        value_batch = [sum(i, []) for i in batch_data[1]]   # (batch_size, num_words)
        src_batch = batch_data[2]                           # (batch_size, num_words)
        tgt_batch = batch_data[3]                           # (batch_size, num_words)

        k_embedding, k_mask = embedding.embed(key_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1)
        v_embedding, v_mask = embedding.embed(value_batch)  # (max_length, batch_size, embed_dim), (max_length, batch_size, 1)
        s_embedding, s_mask = embedding.embed(src_batch)    # (max_length, batch_size, embed_dim), (max_length, batch_size, 1)

        # Encode
        h_k, _ = k_encoder(k_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_v, _ = v_encoder(v_embedding)     # (max_length, batch_size, hidden_size * 2)
        h_s, _ = s_encoder(s_embedding)     # (max_length, batch_size, hidden_size * 2)

        # Decode
        hidden_states = []
        s = torch.zeros(batch_size, hidden_size) # s_0
        y = torch.zeros(batch_size, hidden_size) # y_0
        max_tgt_len = max(len(words) for words in tgt_batch)
        for t in range(1, max_tgt_len + 1):
            alpha = [u_a(torch.tanh(w_a(h_i) + v_a(s))) for h_i in h_s] # (max_length, batch_size, 1)
            alpha = F.softmax(torch.stack(alpha) * s_mask, dim=0)       # (max_length, batch_size, 1)
            ctx = torch.sum(alpha * h_s, dim=0)                         # (batch_size, hidden_size)
            s, y = decoder(ctx, (s, y))                                 # s_t and y_t
            hidden_states.append(s)                                     # (cur_length, batch_size, hidden_size)

        break

if __name__ == "__main__":
    main()
    print("success")
