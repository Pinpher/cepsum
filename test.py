import torch

a = torch.stack([torch.IntTensor([1,2]) for i in range(4)])
mask = torch.IntTensor([[0],[1],[0],[1]]).byte()
#mask = torch.ByteTensor([0,1,0,1])

print(a[:,1])
print(torch.masked_select(a[:,1], mask))
#print(a[:,1] * mask)