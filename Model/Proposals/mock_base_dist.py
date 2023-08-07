
import torch
import torch.nn as nn

class MockBaseDist(nn.Module):
    '''
    Mock base distribution returning 0. 
    '''
    def __init__(self) -> None:
        super().__init__()
        self.mock_param = torch.nn.parameter.Parameter(torch.zeros(1, requires_grad=False))
    
    def log_prob(self, x):
        '''
        Mock log probability returning 0.
        '''
        return torch.zeros((x.shape[0], 1,), dtype=x.dtype, device=x.device)

class MockBaseDistRegression(nn.Module):
    '''
    Mock base distribution returning 0. 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def log_prob(self, x, y):
        '''
        Mock log probability returning 0.
        '''
        return torch.zeros(y.shape[0], 1, dtype=x.dtype, device=x.device)
