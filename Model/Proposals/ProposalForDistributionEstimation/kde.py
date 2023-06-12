import torch.distributions as dist
import numpy as np
import torch
import sklearn.neighbors
from .abstract_proposal import AbstractProposal

class KernelDensity(AbstractProposal):
    '''
    Kernel density estimation using sklearn.
    '''
    def __init__(self, input_size, dataset, kernel='gaussian', bandwith = 'scott', nb_center=1000, **kwargs) -> None:
        super().__init__(input_size=input_size)
        
        data = self.get_data(dataset, nb_center).flatten(1).numpy()
        self.kd = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwith).fit(data)

    def sample_simple(self, nb_sample = 1):
        samples = torch.from_numpy(self.kd.sample((nb_sample,))).reshape(nb_sample, *self.input_size).detach()
        return samples
    
    def log_prob_simple(self, x):
        return torch.from_numpy(self.kd.score_samples(x.flatten(1).detach().numpy())).to(x.device, x.dtype)
    
