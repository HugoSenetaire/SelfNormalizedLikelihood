
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from .gaussian_mixture import GaussianMixtureProposal
import numpy as np
from .abstract_proposal import AbstractAdaptiveProposal


def get_GaussianMixtureAdaptiveProposal(default_proposal, input_size, dataset, cfg,):
    return GaussianMixtureAdaptiveProposal(
                default_proposal,
                input_size,
                dataset,
                cfg.nb_sample_estimate,
                )
                                           

class GaussianMixtureAdaptiveProposal(AbstractAdaptiveProposal):
    '''
    Gaussian mixture proposal with adaptive parameters. Sample from a mixture of gaussian whose means are given by the batch x
    and covariance is estimated at the beginning of the training on a subset of the dataset.

    Attributes:
    ----------
    default_proposal : AbstractProposal
        The default proposal used to sample when the adaptive proposal is not used.
    input_size : tuple
        The size of the input.
    x : torch.Tensor
        The samples used to adapt the proposal.
    std : torch.Tensor
        The standard deviation of the gaussian mixture.
    
    Methods:
    --------
    set_x(x): set the samples to use to adapt the proposal.
    log_prob_adaptive(x): compute the log probability of the proposal with the adaptive parameters.
    sample_adaptive(nb_sample): sample from the proposal with the adaptive parameters.
    '''
    def __init__(self, default_proposal, input_size, dataset, nb_sample_estimate = 10000,) -> None:
        super().__init__(input_size=input_size, default_proposal=default_proposal)
        data = self.get_data(dataset, nb_sample_estimate)
        data += torch.randn_like(data) * 1e-2
        self.std = nn.Parameter(data.std(0).reshape(input_size), requires_grad=False)
        self.x = None
        
        
    def sample_adaptive(self, nb_sample = 1):
        if nb_sample > len(self.x) :
            index_sample = np.random.choice(len(self.x), nb_sample, replace=True)
        else :
            index_sample = np.random.choice(len(self.x), nb_sample, replace=False)
        x_repeat = self.x[index_sample].clone().detach().reshape(nb_sample, *self.input_size)
        samples = torch.randn_like(x_repeat) * self.std.unsqueeze(0).expand(x_repeat.shape) + x_repeat
        return samples
    
    def log_prob_adaptive(self, samples):
        samples_expanded = samples.unsqueeze(1).expand(samples.shape[0], self.x.shape[0], *samples.shape[1:]).flatten(2)
        x_expanded = self.x.unsqueeze(0).expand(samples.shape[0], self.x.shape[0], *self.x.shape[1:]).flatten(2)
        current_std = self.std.unsqueeze(0).unsqueeze(0).expand(samples.shape[0], self.x.shape[0], *self.std.shape).flatten(2)
        log_prob = -0.5 * ((samples_expanded - x_expanded) / current_std)** 2 - 0.5 * np.log(2 * np.pi) - torch.log(current_std)    
        log_prob = log_prob.sum(2)  
        log_prob_mean = log_prob.logsumexp(1) - np.log(self.x.shape[0])
        return log_prob_mean

