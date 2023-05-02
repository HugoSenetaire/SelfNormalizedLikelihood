import torch
import torch.nn as nn
import numpy as np

class fc_energy(nn.Module):
    def __init__(self, input_size = (1,10), dims = [100, 100, 100], activation = None, last_layer_bias = True, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear = [nn.Linear(np.prod(input_size), dims[0]),]
        for dim_in, dim_out in zip(dims[:-1],dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(nn.Linear(dim_in, dim_out))
        self.linear.extend([nn.ReLU(), ])
        self.linear = nn.Sequential(*self.linear)
        self.last_layer = nn.Linear(dims[-1], 1, bias = last_layer_bias)
        self.activation = None


    def forward(self, x):
        x = x.flatten(1)
        out = self.linear(x)
        out = self.last_layer(out)
        if self.activation is not None :
            out = self.activation(out)
        return out.reshape(-1,1)
    

