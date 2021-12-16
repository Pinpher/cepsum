import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from myvocabulary import *
from myembedding import *
from mydataset import *
from mymodel import *

batch_size = 32
embed_dim = 300
hidden_size = 128
candidate_size = 1292610
learning_rate = 1e-3
epoch_num = 20

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    for epoch in range(1, epoch_num + 1):
        losses = []
        for i_batch, batch_data in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(batch_data)
            loss.backward()
            optimizer.step()
            losses.append(loss.tolist())

            if (i_batch + 1) % 100 == 0:
                print("Epoch %d Batch %d, train loss %f" % (epoch, i_batch, np.mean(losses[-100:])))
    
if __name__ == "__main__":
    main()
    print("success")
