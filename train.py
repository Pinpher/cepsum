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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: " + str(device))
    model = Mymodel(
        batch_size = batch_size,
        embed_dim = embed_dim,
        hidden_size = hidden_size,
        candidate_size = candidate_size,
        device = device
    )
    model.to(device)

    for i_batch, batch_data in enumerate(dataloader):
        loss = model(batch_data)
        break
    
if __name__ == "__main__":
    main()
    print("success")
