
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from .gmm_torch.gmm import GaussianMixture


class GaussianMixtureProposal(nn.Module):
    def __init__(self, input_size, dataset, covariance_type="diag", eps=1.e-6, n_components = 10, nb_sample_for_estimate = 10000, init_params="kmeans", delta = 1e-3, n_iter = 100, warm_start = False, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.delta = delta
        self.n_iter = n_iter
        self.warm_start = warm_start

        n_features = np.prod(input_size)
        index = np.random.choice(len(dataset), min(nb_sample_for_estimate,len(dataset)))
        data = torch.cat([dataset[i][0] for i in index]).flatten(1)
        data += torch.randn_like(data) * 1e-2
        
        self.gmm = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type=covariance_type, eps=eps, init_params=init_params)
        self.gmm.fit(data, delta=self.delta, n_iter=self.n_iter, warm_start=self.warm_start)
        
    def sample(self, nb_sample = 1):
        samples,y = self.gmm.sample(nb_sample)
        samples = samples.reshape(nb_sample, *self.input_size)
        return samples
    
    def log_prob(self, x):
        sample = x.flatten(1)
        log_prob = self.gmm.score_samples(sample)
        return log_prob.reshape(x.shape[0],)

