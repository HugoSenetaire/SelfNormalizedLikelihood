import torch
import torch.nn as nn


class MockBias(nn.Module):
    '''
    Mock bias returning 0.
    '''
    def __init__(self, input_size_x_feature=1, ):
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    
    def forward(self, x):
        return x
    
class ScalarBias(nn.Module):
    '''
    Scalar bias for distribution estimation.
    '''
    def __init__(self, input_size_x_feature=1, ):
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    
    def forward(self, x):
        return x + self.bias.expand((x.shape[0], 1))