
import torch
import torch.nn as nn
import numpy as np
from .abstract_proposal import AbstractAdaptiveProposal


def get_NoiseGradationAdaptiveProposal(default_proposal, input_size, dataset, cfg,):
    return NoiseGradationAdaptiveProposal(
                default_proposal,
                input_size,
                dataset,
                cfg.ranges_std,
                cfg.nb_sample_estimate,
                )

class NoiseGradationAdaptiveProposal(AbstractAdaptiveProposal):
    '''
    Gaussian mixture adaptive but with different gradation of noise controlled by range std.
    The base std is calculated with the data and then we multiply it by the range std.
    '''
    def __init__(self, default_proposal, input_size, dataset, ranges_std = [0.1, 1.,], nb_sample_estimate = 10000, **kwargs) -> None:
        super().__init__(input_size=input_size, default_proposal=default_proposal)
        data = self.get_data(dataset, nb_sample_estimate).reshape(-1, np.prod(self.input_size))
        data += torch.randn_like(data) * 1e-2
        self.ranges_std = ranges_std
        self.nb_gradation = len(self.ranges_std)
        
        std_aux = []
        current_std = data.std(0)
        for k in range(self.nb_gradation):
            std_aux = std_aux + [current_std.clone() * self.ranges_std[k]]
        self.std = nn.Parameter(torch.stack(std_aux), requires_grad=True)
        

        
    def sample_adaptive(self, nb_sample = 1):
        batch_size = self.x.shape[0]
        index_std = np.random.choice(self.nb_gradation, nb_sample)
        if nb_sample > len(self.x) :
            index_samples = np.random.choice(len(self.x), nb_sample, replace=True)
        else :
            index_samples = np.random.choice(len(self.x), nb_sample, replace=False)

        x_repeat = self.x[index_samples].clone().detach().reshape(nb_sample, *self.input_size)
        samples = torch.randn_like(x_repeat) * self.std[index_std].reshape(x_repeat.shape) + x_repeat
        return samples
    
    def log_prob_adaptive(self, samples):
        samples_expanded = samples.unsqueeze(0).unsqueeze(2).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *samples.shape[1:]).flatten(3)
        x_expanded = self.x.unsqueeze(0).unsqueeze(0).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *self.x.shape[1:]).flatten(3)
        current_std = self.std.unsqueeze(1).unsqueeze(1).expand(self.std.shape[0], samples.shape[0], self.x.shape[0], *self.std.shape[1:]).flatten(3)
        log_prob = -0.5 * ((samples_expanded - x_expanded) / current_std)** 2 - 0.5 * np.log(2 * np.pi) - torch.log(current_std)    
        log_prob = log_prob.flatten(3).sum(3)  
        log_prob_mean = log_prob.logsumexp(2).logsumexp(0) - np.log(self.x.shape[0]*self.nb_gradation)
        return log_prob_mean

