import torch.distributions as dist
import torch.nn as nn
import numpy as np
import torch
import sklearn.neighbors
from ..gmm_torch import GaussianMixture


class KernelDensityAdaptive(nn.Module):
    def __init__(self, input_size, dataset, kernel='gaussian', bandwith = 'scott', nb_center=1000, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        if isinstance(dataset, list):
            dataset = dataset[0]
        
        index = np.random.choice(len(dataset), min(nb_center,len(dataset)))
        data = torch.cat([dataset[i][0] for i in index]).flatten(1).numpy()
        self.kd = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwith).fit(data)
        self.bandwith = torch.nn.Parameter(torch.tensor(self.kd.bandwidth_,), requires_grad=False)
        self.last_x = None

    def set_x(self, x):
        if self.training :
            self.last_x = x
        else :
            self.last_x = None


    def sample(self, nb_sample = 1, ):
        if self.last_x is None :
            samples = torch.from_numpy(self.kd.sample((nb_sample,)),)
        else :
            print(self.last_x.shape)
            batch_size = self.last_x.shape[0]
            if batch_size >= nb_sample :
                index = np.random.choice(len(self.last_x), nb_sample, replace=False)
            else :
                index = np.random.choice(len(self.last_x), nb_sample, replace=True)
            current_mu = self.last_x.detach()
            current_sigma = torch.full_like(current_mu, self.bandwith)
            current_mu = current_mu[index]
            current_sigma = current_sigma[index]
            samples = current_mu + current_sigma * torch.randn_like(current_mu)
        return samples.detach().reshape(nb_sample, *self.input_size)
    
    def log_prob(self, x):
        # if self.last_x is None :
        
        return torch.from_numpy(self.kd.score_samples(x.flatten(1).numpy()))
        # else :
        #     x_flatten = x.reshape(-1, 1, *self.input_size).flatten(2)
        #     nb_center = self.last_x.shape[0]
        #     current_mu = self.last_x.detach().unsqueeze(0).flatten(2)
        #     current_sigma = torch.full_like(current_mu, self.bandwith).reshape(1, nb_center, *self.input_size).flatten(2)
        #     weights = torch.full((1,nb_center), 1.0)

        #     prec = torch.rsqrt(current_sigma)

        #     log_p = torch.sum((current_mu * current_mu + x_flatten * x_flatten - 2 * x_flatten * current_mu) * prec, dim=2, keepdim=True)
        #     log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)
        #     log_prob = (weights * (log_p - log_det).sum(-1)).sum(-1)
        #     return log_prob
