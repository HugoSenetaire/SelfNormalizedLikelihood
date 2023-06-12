import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Layer1FC(nn.Module):
    def __init__(self, input_size_x_feature) -> None:
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.fc1 = nn.Linear(np.prod(input_size_x_feature), 1)
    
    def forward(self, x):
        x = x.flatten(1)
        return self.fc1(x).reshape(-1,1)
    
class Layer2FC(nn.Module):
    def __init__(self, input_size_x_feature, hidden_dim = 10) -> None:
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.fc1 = nn.Linear(np.prod(input_size_x_feature), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).reshape(-1,1)

class Layer3FC(nn.Module):
    def __init__(self, input_size_x_feature, hidden_dim = 10) -> None:
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.fc1 = nn.Linear(np.prod(input_size_x_feature), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).reshape(-1,1)