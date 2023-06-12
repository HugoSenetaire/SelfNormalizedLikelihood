import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class MDNProposal_Network(nn.Module):
    def __init__(self, hidden_dim, K = 4, input_size_y = 1, **kwargs):
        super().__init__()

        self.K = K
        self.input_size_y = input_size_y
        self.dim_input_y = np.prod(input_size_y)
        self.hidden_dim = max(hidden_dim,10)
        

        self.fc1_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, self.dim_input_y*self.K)

        self.fc1_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, self.dim_input_y*self.K)

        self.fc1_weight = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, self.K)


    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))
        means = F.relu(self.fc1_mean(x_feature))  # (shape: (batch_size, hidden_dim))
        means = self.fc2_mean(means)  # (shape: batch_size, K))

        log_sigma2s = F.relu(self.fc1_sigma(x_feature))  # (shape: (batch_size, hidden_dim))
        log_sigma2s = self.fc2_sigma(log_sigma2s)  # (shape: batch_size, K))

        weight_logits = F.relu(self.fc1_weight(x_feature))  # (shape: (batch_size, hidden_dim))
        weight_logits = self.fc2_weight(weight_logits)  # (shape: batch_size, K))
        weights = torch.softmax(weight_logits, dim=1) # (shape: batch_size, K))
        return means, log_sigma2s, weights


        