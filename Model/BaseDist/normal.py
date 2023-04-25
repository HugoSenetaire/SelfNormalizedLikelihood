
import torch.nn as nn
import torch.distributions as distributions
import torch



class Normal(nn.Module):
    def __init__(self, dim, mu = None, logstd = None, learn_mu = False, learn_logstd = False, **kwargs):
        super(Normal, self).__init__()
        if mu is None:
            mu = torch.zeros(dim)
        if logstd is None:
            logstd = torch.zeros(dim)
        
        self.mu = nn.Parameter(mu, requires_grad=learn_mu)
        self.logstd = nn.Parameter(logstd, requires_grad=learn_logstd)
        self.dim = dim

    def sample(self, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        return torch.randn(nb_sample, *self.dim) * self.logstd.exp() + self.mu
    
    def log_prob(self, x):
        '''
        Calculate energy of the samples from the energy function.
        '''
        return distributions.Normal(self.mu, self.logstd.exp()).log_prob(x).view(x.size(0), -1).sum(1)