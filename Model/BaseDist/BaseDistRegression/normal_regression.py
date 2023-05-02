
import torch.nn as nn
import torch.distributions as distributions
import torch
import numpy as np




class NormalRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, mode_cov = 'diag', learn_mu = False, learn_logstd = False,  **kwargs) -> None:
        super().__init__()
        # assert False
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        
        index = np.random.choice(len(dataset), 1000)
        data = torch.cat([dataset[i][1] for i in index])
        self.mode_cov = mode_cov

        if mode_cov == 'diag':
            self.logstd = data.std(0).log()
            self.mu = data.mean(0)
        elif mode_cov == 'spherical' :
            self.mu = data.mean()
            self.logstd = data.std().log()

 
        self.mu = nn.Parameter(self.mu, requires_grad=learn_mu)
        self.logstd = nn.Parameter(self.logstd, requires_grad=learn_logstd)


    def sample(self, x_feature, nb_sample = 1):
        batch_size = x_feature.shape[0]
        distribution = distributions.Normal(self.mu, self.logstd.exp())

        if self.mode_cov == 'diag':
            samples = distribution.rsample((batch_size, nb_sample,)).reshape(batch_size, nb_sample,*self.input_size_y)
        elif self.mode_cov == 'spherical' :
            samples = distribution.rsample((batch_size, nb_sample,*self.input_size_y))
        return samples
    
    def log_prob(self, x_feature, y):
        assert x_feature.shape[0] == y.shape[0]

        distribution = distributions.Normal(self.mu, self.logstd.exp())
        if self.mode_cov == 'diag':
            log_p = distribution.log_prob(y).flatten(1).sum(1)
        elif self.mode_cov == 'spherical' :
            y_flatten = y.flatten()
            log_p = distribution.log_prob(y_flatten)
            log_p = log_p.reshape(y.shape[0],-1).sum(1)
        return log_p
    
    