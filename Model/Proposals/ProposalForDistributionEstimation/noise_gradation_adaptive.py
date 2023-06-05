
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from .gaussian_mixture import GaussianMixtureProposal
import numpy as np


class NoiseGradationAdaptiveProposal(nn.Module):
    def __init__(self, default_proposal, input_size, dataset, nb_gradation = 3, covariance_type="diag", std = 'dataset', nb_sample_for_estimate = 10000, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.default_proposal = default_proposal

        n_features = np.prod(input_size)
        index = np.random.choice(len(dataset), min(nb_sample_for_estimate,len(dataset)))
        data = torch.cat([dataset[i][0] for i in index]).reshape(len(index), -1)
        data += torch.randn_like(data) * 1e-2
        self.ranges_std = [0.1, 1,]
        self.nb_gradation = len(self.ranges_std)
        
        std_aux = []
        current_std = data.std(0)
        for k in range(self.nb_gradation):
            std_aux = std_aux + [current_std.clone() * self.ranges_std[k]]
        self.std = nn.Parameter(torch.stack(std_aux), requires_grad=False)
        self.x = None
        
    def set_x(self, x):
        self.x = x
        
    def sample(self, nb_sample = 1):
        if self.x is None or not self.training:
            samples = self.default_proposal.sample(nb_sample)
            return samples
        else :
            # print(self.x)
            batch_size = self.x.shape[0]
            index_std = np.random.choice(self.nb_gradation, nb_sample)
            if nb_sample > len(self.x) :
                index_samples = np.random.choice(len(self.x), nb_sample, replace=True)
            else :
                index_samples = np.random.choice(len(self.x), nb_sample, replace=False)

            x_repeat = self.x[index_samples].clone().detach().reshape(nb_sample, *self.input_size)
         
            samples = torch.randn_like(x_repeat) * self.std[index_std].reshape(x_repeat.shape) + x_repeat
            # index = np.random.choice(len(samples), nb_sample)
            return samples
    
    def log_prob(self, samples):
        if self.x is None or not self.training:
            return self.default_proposal.log_prob(samples)
        else :
            samples_expanded = samples.unsqueeze(0).unsqueeze(2).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *samples.shape[1:]).flatten(3)
            x_expanded = self.x.unsqueeze(0).unsqueeze(0).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *self.x.shape[1:]).flatten(3)
            current_std = self.std.unsqueeze(1).unsqueeze(1).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *self.std.shape[1:]).flatten(3)
            log_prob = -0.5 * ((samples_expanded - x_expanded) / current_std)** 2 - 0.5 * np.log(2 * np.pi) - torch.log(current_std)    
            log_prob = log_prob.flatten(3).sum(3)  
            log_prob_mean = log_prob.logsumexp(2).logsumexp(0) - np.log(self.x.shape[0]*self.nb_gradation)
            return log_prob_mean

