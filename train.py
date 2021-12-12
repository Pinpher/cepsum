from torch.utils.data import DataLoader
from mydataset import *

data = MyDataset("data/cut_valid.txt")
dataloader = DataLoader(data, batch_size=32, collate_fn=collate_fn)
for i_batch, batch_data in enumerate(dataloader):
    print(batch_data)
    break
