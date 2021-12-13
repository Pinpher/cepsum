from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from mydataset import *
from Bi_LSTM import *

data = MyDataset("data/cut_valid.txt")
dataloader = DataLoader(data, batch_size=32, collate_fn=collate_fn)

embed_dim = 300
batch_size = 32

# padding for input sectence
def padding(x):
    max_len = 0
    for sentence in x:
        max_len = max(len(sentence), max_len)
    for i, sentence in enumerate(x):
        pad = torch.zeros([max_len - len(sentence), embed_dim])
        x[i] = torch.cat((sentence, pad), dim=0)
    return torch.stack(x)

def main():
    for i_batch, batch_data in enumerate(dataloader):
        # batch_data: 4-tuple (key_batch, value_batch, src_batch, tgt_batch)
        # key_batch & value_batch: (batch_size, 
        #                           attribute list length,      # unfixed 
        #                           word num of key / value,    # unfixed 
        #                           embedd_dim)
        # src_batch & tgt_batch: (batch_size, 
        #                         word num of src / tgt,        # unfixed 
        #                         embedd_dim)

        #reshape & padding
        key_batch = [torch.stack([j for i in batch_data[0][l] for j in i]).squeeze(1) for l in range(batch_size)]
        value_batch = [torch.stack([j for i in batch_data[1][l] for j in i]).squeeze(1) for l in range(batch_size)]
        src_batch = [i.squeeze(1) for i in batch_data[2]]
        tgt_batch = [i.squeeze(1) for i in batch_data[3]]

        model1 = Bi_LSTM(input_size=embed_dim, hidden_size=128, batch_size=batch_size)
        key_h = model1(padding(key_batch))

        model2 = Bi_LSTM(input_size=embed_dim, hidden_size=128, batch_size=batch_size)
        tgt_h = model1(padding(tgt_batch))

        break

if __name__ == "__main__":
    main()
