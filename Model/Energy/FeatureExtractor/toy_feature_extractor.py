import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ToyFeatureNet(nn.Module):
    '''
    Toy feature extractor returning the input, same implementation as the paper
    Learning Proposal for Energy Based Regression.
    '''
    def __init__(self, input_size_x=1, hidden_dim=10):
        super().__init__()
        self.input_dim = np.prod(input_size_x)
        self.fc1_x = nn.Linear(self.input_dim, hidden_dim)
        self.fc2_x = nn.Linear(hidden_dim, hidden_dim)
        self.output_size = hidden_dim

    def forward(self, x):
        # (x has shape (batch_size, 1))
        x = x.flatten(1)
        x_feature = F.relu(self.fc1_x(x)) # (shape: (batch_size, hidden_dim))
        x_feature = F.relu(self.fc2_x(x_feature)) # (shape: (batch_size, hidden_dim))

        return x_feature.reshape(-1, 1, self.output_size) # (shape: (batch_size, 1, hidden_dim)
