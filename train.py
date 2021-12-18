import os
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from mydataset import *
from mymodel import *

batch_size = 32
embed_dim = 300
hidden_size = 512
learning_rate = 2e-3
epoch_num = 20
model_save_path = "./model"
name = "test"

train_data = MyDataset("data/cut_train.txt")
valid_data = MyDataset("data/cut_valid.txt")
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: " + str(device))
    model = Mymodel(
        batch_size = batch_size,
        embed_dim = embed_dim,
        hidden_size = hidden_size,
        device = device
    )
    model = nn.DataParallel(model.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    for epoch in range(1, epoch_num + 1):
        start_time = time.time()
        train_losses = []
        for i_batch, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, _ = model(batch_data)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.tolist())

            if i_batch % 10 == 0:
                print("Epoch %d Batch %d, train loss %f" % (epoch, i_batch, np.mean(train_losses[-10:])), flush=True)

        valid_losses = []
        model.eval()
        for i_batch, batch_data in enumerate(valid_dataloader):
            loss, _ = model(batch_data)
            valid_losses.append(loss.mean().detach().cpu())
        model.train()
        print("Epoch " + str(epoch) + " finished, took " + str(time.time() - start_time) + "s", flush=True)
        print("valid loss " + str(np.mean(valid_losses)), flush=True)

        path = os.path.join(model_save_path, "model_%s_%d" % (name, epoch))
        torch.save(model.module.state_dict(), path)
        torch.save(model.module.embedding.embedding.state_dict(), path + "_embedding")

if __name__ == "__main__":
    main()
