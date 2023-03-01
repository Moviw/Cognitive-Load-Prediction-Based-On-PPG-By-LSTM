import numpy as np
import torch


def get_sequence(data, seq_length, device, to_device=True):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length), :-1]
        _y = data[i+seq_length, -1]
        x.append(_x)
        y.append(_y)

    if to_device:
        return torch.Tensor(np.array(x)).to(device), torch.Tensor(np.array(y)).to(device)
    else:
        return np.array(x), np.array(y).reshape(-1,1)