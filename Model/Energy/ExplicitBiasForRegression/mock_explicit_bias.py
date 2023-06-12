import torch.nn as nn
import torch 

class MockBiasRegression(nn.Module):
    '''
    Mock bias regression returning the input.
    '''
    def __init__(self, input_size_x_feature=1, ):
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
    
    def forward(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)