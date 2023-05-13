
import torch.nn as nn
import torch.distributions as distributions
import torch
import numpy as np




class Normal(nn.Module):
    def __init__(self, input_size, dataset, mode_cov = 'diag', learn_mu = False, learn_logstd = False,  **kwargs) -> None:
        super().__init__()
        # assert False
        self.input_size = input_size
        if isinstance(dataset, list):
            dataset = dataset[0]
        
        index = np.random.choice(len(dataset), 1000)
        data = torch.cat([dataset[i][0] for i in index]).reshape(-1, *input_size)
        self.mode_cov = mode_cov

        if mode_cov == 'diag':
            self.logstd = data.std(0).log()
            self.mu = data.mean(0)
        elif mode_cov == 'spherical' :
            self.mu = data.mean()
            self.logstd = data.std().log()

 
        self.mu = nn.Parameter(self.mu, requires_grad=learn_mu)
        self.logstd = nn.Parameter(self.logstd, requires_grad=learn_logstd)


    def sample(self, nb_sample = 1):
        distribution = distributions.Normal(self.mu, self.logstd.exp())

        if self.mode_cov == 'diag':
            samples = distribution.rsample((nb_sample,)).reshape(nb_sample,*self.input_size)
        elif self.mode_cov == 'spherical' :
            samples = distribution.rsample((nb_sample,*self.input_size))
        return samples
    
    def log_prob(self, x):
        x = x.reshape(x.shape[0],*self.input_size)
        distribution = distributions.Normal(self.mu, self.logstd.exp())
        if self.mode_cov == 'diag':
            log_p = distribution.log_prob(x).flatten(1).sum(1)
        elif self.mode_cov == 'spherical' :
            x_flatten = x.flatten()
            log_p = distribution.log_prob(x_flatten)
            log_p = log_p.reshape(x.shape[0],-1).sum(1)
        # print(log_p.shape)
        return log_p
    
    