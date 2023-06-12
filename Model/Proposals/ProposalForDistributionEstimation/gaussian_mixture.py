
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from ..gmm_torch.gmm import GaussianMixture
from .abstract_proposal import AbstractProposal

class GaussianMixtureProposal(AbstractProposal):
    '''
    Gaussian mixture proposal.

    Attributes:
    ----------
    input_size : tuple
        The size of the input.
    gmm : GaussianMixture
        The gaussian mixture model.
    n_components : int
        The number of components of the gaussian mixture model.
    delta : float
        The delta parameter of the gaussian mixture model.
    n_iter : int
        The number of iterations to fit the gaussian mixture model.
    warm_start : bool
        Whether to warm start the gaussian mixture model.
    covariance_type : str
        The type of covariance to use for the gaussian mixture model (diag, full, spherical).
    eps : float
        The epsilon parameter of the gaussian mixture model.
    init_parameters : str
        The type of initialization to use for the gaussian mixture model (kmeans, random).
    nb_sample_for_estimate : int
        The number of samples to use to estimate the gaussian mixture model.

    Methods:
    --------
    log_prob_simple(x): compute the log probability of the proposal.
    sample_simple(nb_sample): sample from the proposal.
    '''

    def __init__(self, input_size, dataset, covariance_type="diag", eps=1.e-6, n_components = 10, nb_sample_for_estimate = 10000, init_parameters="kmeans", delta = 1e-3, n_iter = 100, warm_start = False, fit = True, **kwargs) -> None:
        super().__init__(input_size=input_size)
        self.n_components = n_components
        self.delta = delta
        self.n_iter = n_iter
        self.warm_start = warm_start

        n_features = np.prod(input_size)
        data = self.get_data(dataset, nb_sample_for_estimate)
        data += torch.randn_like(data) * 1e-2
        
        self.gmm = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type=covariance_type, eps=eps, init_parameters=init_parameters)
        for param in self.gmm.parameters():
            param.requires_grad = True
        if fit :
            self.gmm.fit(data, delta=self.delta, n_iter=self.n_iter, warm_start=self.warm_start)
        
    def sample_simple(self, nb_sample = 1):
        samples, y = self.gmm.sample(nb_sample)
        return samples
    
    def log_prob_simple(self, x):
        sample = x.flatten(1)
        log_prob = self.gmm.score_samples(sample)
        return log_prob

