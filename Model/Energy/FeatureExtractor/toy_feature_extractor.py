import torch.nn as nn
import torch.nn.functional as F


class ToyFeatureNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc2_x = nn.Linear(hidden_dim, hidden_dim)
        self.output_size = hidden_dim

    def forward(self, x):
        # (x has shape (batch_size, 1))
        x = x.flatten(1)
        x_feature = F.relu(self.fc1_x(x)) # (shape: (batch_size, hidden_dim))
        x_feature = F.relu(self.fc2_x(x_feature)) # (shape: (batch_size, hidden_dim))

        return x_feature
