import os
import time
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from mydataset import *
from mymodel import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="only_copy")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--epoch_num', type=int, default=10)
parser.add_argument('--model_save_path', type=str, default="./model")
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--attri_words_path', type=str, default='./vocab/attr_words.txt')
args = parser.parse_args()

batch_size = args.batch_size
embed_dim = args.embed_dim
hidden_size = args.hidden_size
learning_rate = args.learning_rate
epoch_num = args.epoch_num
model_save_path = args.model_save_path
attri_words_path = args.attri_words_path
name = args.name

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
        device = device,
        attri_words_path = attri_words_path
    )

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        model.embedding.embedding.load_state_dict(torch.load(args.resume + "_embedding"))
        model.embedding_tgt.embedding.load_state_dict(torch.load(args.resume + "_embedding_tgt"))

    model = model.to(device)
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
        torch.save(model.state_dict(), path)
        torch.save(model.embedding.embedding.state_dict(), path + "_embedding")
        torch.save(model.embedding_tgt.embedding.state_dict(), path + "_embedding_tgt")

if __name__ == "__main__":
    main()
