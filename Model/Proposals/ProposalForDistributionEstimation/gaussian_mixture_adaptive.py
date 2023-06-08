
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from .gaussian_mixture import GaussianMixtureProposal
import numpy as np


class GaussianMixtureAdaptiveProposal(nn.Module):
    def __init__(self,
                default_proposal,
                input_size,
                dataset,
                covariance_type="diag",
                std = 'dataset',
                nb_sample_for_estimate = 10000,
                feature_extractor = None,
                **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.default_proposal = default_proposal

        n_features = np.prod(input_size)
        index = np.random.choice(len(dataset), min(nb_sample_for_estimate,len(dataset)))
        with torch.no_grad():
            if feature_extractor is None :
                data = torch.cat([dataset[i][0] for i in index]).reshape(len(index), *input_size)
            else :
                data = torch.cat([feature_extractor(dataset[i][0].unsqueeze(0)) for i in index]).reshape(len(index), -1)
        data += torch.randn_like(data) * 1e-2
        self.std = nn.Parameter(data.std(0).reshape(input_size), requires_grad=False)
        self.x = None
        
    def set_x(self, x):
        self.x = x
        
    def sample(self, nb_sample = 1):
        if self.x is None or not self.training:
            samples = self.default_proposal.sample(nb_sample)
            return samples
        else :
            batch_size = self.x.shape[0]
            if nb_sample > len(self.x) :
                index_sample = np.random.choice(len(self.x), nb_sample, replace=True)
            else :
                index_sample = np.random.choice(len(self.x), nb_sample, replace=False)
            x_repeat = self.x[index_sample].clone().detach().reshape(nb_sample, *self.input_size)
            samples = torch.randn_like(x_repeat) * self.std.unsqueeze(0).expand(x_repeat.shape) + x_repeat
            return samples.detach()
    
    def log_prob(self, samples):
        if self.x is None or not self.training:
            return self.default_proposal.log_prob(samples)
        else :
            samples_expanded = samples.unsqueeze(1).expand(samples.shape[0], self.x.shape[0], *samples.shape[1:]).flatten(2)
            x_expanded = self.x.unsqueeze(0).expand(samples.shape[0], self.x.shape[0], *self.x.shape[1:]).flatten(2)
            current_std = self.std.unsqueeze(0).unsqueeze(0).expand(samples.shape[0], self.x.shape[0], *self.std.shape).flatten(2)
            log_prob = -0.5 * ((samples_expanded - x_expanded) / current_std)** 2 - 0.5 * np.log(2 * np.pi) - torch.log(current_std)    
            log_prob = log_prob.sum(2)  
            log_prob_mean = log_prob.logsumexp(1) - np.log(self.x.shape[0])
            return log_prob_mean

