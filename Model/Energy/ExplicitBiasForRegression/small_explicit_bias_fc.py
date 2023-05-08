import torch.nn as nn
import torch.nn.functional as F

class Layer1FC(nn.Module):
    def __init__(self, input_size_x) -> None:
        super().__init__()
        self.input_size_x = input_size_x
        self.fc1 = nn.Linear(input_size_x, 1)
    
    def forward(self, x):
        return self.fc1(x)
    
class Layer2FC(nn.Module):
    def __init__(self, input_size_x, hidden_dim = 10) -> None:
        super().__init__()
        self.input_size_x = input_size_x
        self.fc1 = nn.Linear(input_size_x, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Layer3FC(nn.Module):
    def __init__(self, input_size_x, hidden_dim = 10) -> None:
        super().__init__()
        self.input_size_x = input_size_x
        self.fc1 = nn.Linear(input_size_x, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)