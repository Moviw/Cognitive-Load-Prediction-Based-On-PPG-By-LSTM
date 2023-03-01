import torch
from torch import nn
import numpy as np


class model(nn.Module):

    '''
        Parameters:
            input_size :输入维度,对于PPG信号来说就是2 1个是sVRI 另一个是压力状态
            hidden_size:隐藏层数
            num_layers :多少个lstm层堆叠

        Inputs: input, (h_0, c_0)
            input:
                三维矩阵(样本个数batch_size,单个样本序列长度sequence length,输入维度input_size)

            h_0;
                三维矩阵(多少层lstm堆叠num_layers,样本个数batch_size,输入维度input_size)
                初始全0短期记忆矩阵

            c_0:
                三维矩阵(多少层lstm堆叠num_layers,样本个数batch_size,输入维度input_size)
                初始全0长期记忆矩阵

        Outputs: output, (h_n, c_n)
            output:
                三维矩阵(样本个数batch_size,单个样本序列长度sequence length,隐藏层数hidden_size)
                保存了最后一层,每个time step的输出h

            h_n: 
                三维矩阵(多少层lstm堆叠num_layers,样本个数batch_size,隐藏层数hidden_size)
                保存了每一层,最后一个time step的输出h

            c_n: 
                三维矩阵(多少层lstm堆叠num_layers,样本个数batch_size,隐藏层数hidden_size)
                最后一次长期记忆
                保存了每一层,最后一个time step的输出c
    '''

    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super(model, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size).to(self.device)

        c_0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size).to(self.device)

        output, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        output = output[:, -1, :].view(-1, self.hidden_size)

        output = self.fc1(output)
        out = self.fc2(output)
        out = self.softmax(out)

        return out
