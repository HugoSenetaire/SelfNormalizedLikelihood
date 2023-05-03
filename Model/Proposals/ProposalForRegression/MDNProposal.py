
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os


class MDNProposal_Network(nn.Module):
    def __init__(self, hidden_dim, K = 4, input_size_y = 1, **kwargs):
        super().__init__()

        self.K = K
        self.input_size_y = input_size_y
        self.hidden_dim = hidden_dim

        self.fc1_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, self.input_size_y*self.K)

        self.fc1_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, self.input_size_y*self.K)

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
    
        
class MDNProposalRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, K = 4,):
        super().__init__()
        self.K = K
        self.input_size_x = np.prod(input_size_x)
        self.input_size_y = np.prod(input_size_y)
        self.network = MDNProposal_Network(self.input_size_x, K, self.input_size_y)



    def sample(self, x_feature, nb_sample = 1, ):
        batch_size = x_feature.size(0)

        means, log_sigma2s, weights = self.network(x_feature)
        sigmas = torch.exp(log_sigma2s/2.0)
        means = means.view(-1, self.input_size_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        sigmas = sigmas.view(-1, self.input_size_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        y_samples_K = q_distr.sample(sample_shape=torch.Size([nb_sample])) # (shape: (num_samples, batch_size, self.input_size_y, K))
        inds = torch.multinomial(weights, num_samples=nb_sample, replacement=True).unsqueeze(2).unsqueeze(2) # (shape: (batch_size, num_samples, 1, 1))
        inds = inds.expand(batch_size, nb_sample, self.input_size_y, 1) # (shape: (batch_size, num_samples, self.input_size_y, 1))
        inds = torch.transpose(inds, 1, 0) # (shape: (num_samples, batch_size, self.input_size_y, 1))
        y_samples = y_samples_K.gather(3, inds).squeeze(3) # (shape: (num_samples, batch_size, self.input_size_y))
        y_samples = y_samples.detach()
        y_samples = torch.transpose(y_samples, 1, 0).reshape(batch_size, nb_sample, self.input_size_y) # (shape: (batch_size, num_samples, input_size_y ))



    def log_prob(self, x_feature, y):
        batch_size = x_feature.size(0)
        means, log_sigma2s, weights = self.network(x_feature)
        sigmas = torch.exp(log_sigma2s/2.0)
        means = means.view(batch_size, self.input_size_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        sigmas = sigmas.view(batch_size, self.input_size_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        y_expanded = y.unsqueeze(-1).expand(batch_size, self.input_size_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        log_q_ys_K = q_distr.log_prob(y_expanded).sum(1) # (shape: (batch_size, K)
        log_q_ys = torch.logsumexp(torch.log(weights) + log_q_ys_K, dim=1) # (shape: (batch_size))
        return log_q_ys

