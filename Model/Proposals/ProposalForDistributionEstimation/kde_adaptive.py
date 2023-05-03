import torch.distributions as dist
import numpy as np
import torch
import sklearn.neighbors
from gmm_torch import GaussianMixture


class KernelDensityAdaptive():
    def __init__(self, input_size, dataset, kernel='gaussian', bandwith = 'scott', nb_center=1000, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        
        index = np.random.choice(len(dataset), min(nb_center,len(dataset)))
        data = torch.cat([dataset[i][0] for i in index]).flatten(1).numpy()
        self.kd = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwith).fit(data)
        self.bandwith = self.kd.bandwidth_


    def sample(self, nb_sample = 1, input_x = None):
        if input_x is None :
            samples = torch.from_numpy(self.kd.sample((nb_sample,))).reshape(nb_sample, *self.input_size).detach()
        else :
            current_x = input_x.detach().clone()
            
            # current_x = current_x.flatten(1).numpy()
        return samples
    
    def log_prob(self, x):
        # return torch.from_numpy(self.kd.score_samples(x.flatten(1).detach().numpy())).to(x.device, x.dtype)
    
