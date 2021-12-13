import torch
import logging
from torch import nn
import torch.nn.functional as F

class Bi_LSTM(nn.Module):
    def __init__(self, 
            input_size,             # embed_dim
            hidden_size,            # h_dim
            batch_size):           
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # fw & bw use different cells
        self.lstmCell_fw = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=False)
        self.lstmCell_bw = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=False)
        return
    
    def forward(self, x):
        # x : (bach_size, sentence_length, embed_dim)
        sentence_length, embed_dim = x[0].shape
        # forward
        h_fw, c_fw = self.init()
        h_fw_each_step = []
        for i in range(sentence_length):
            x_i = x[:, i]                                     #(batch_size, 1, embed_dim)
            h_fw, c_fw = self.lstmCell_fw(x_i, (h_fw, c_fw))  #(batch_size, embed_dim)
            h_fw_each_step.append(h_fw)
        # backword
        h_bw, c_bw = self.init()
        h_bw_each_step = []
        for i in range(sentence_length):
            x_i = x[:, sentence_length - 1 - i]               #(batch_size, 1, embed_dim)
            h_bw, c_bw = self.lstmCell_bw(x_i, (h_bw, c_bw))  #(batch_size, embed_dim)
            h_bw_each_step.append(h_bw)
        # add & output
        h_bw_each_step.reverse()
        h_fw_each_step = torch.stack(h_fw_each_step, dim=0)   #(sentence_length, batch_size, embed_dim)
        h_bw_each_step = torch.stack(h_bw_each_step, dim=0)   #(sentence_length, batch_size, embed_dim)
        return h_fw_each_step + h_bw_each_step
    
    def init(self):
        # (batch_size, hidden_dim)
        h0 = torch.zeros(self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.batch_size, self.hidden_size)
        return h0, c0