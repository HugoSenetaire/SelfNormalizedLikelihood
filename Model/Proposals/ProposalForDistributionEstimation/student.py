import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn
from .abstract_proposal import AbstractProposal

class StudentProposal(AbstractProposal):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__(input_size=input_size)

        data = self.get_data(dataset, 10000).reshape(-1, *self.input_size)
        self.mean = nn.parameter.Parameter(data.mean(0), requires_grad=True)
        self.log_std = nn.parameter.Parameter(data.std(0).log(), requires_grad=True)
        self.log_df = nn.parameter.Parameter(torch.zeros_like(self.mean.data), requires_grad=True)


    def sample_simple(self, nb_sample = 1):
        self.distribution = dist.StudentT(self.log_df.exp(), self.mean, self.log_std.exp())
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, *self.input_size)
        return samples
    
    def log_prob_simple(self, x):
        self.distribution = dist.StudentT(self.log_df.exp(), self.mean, self.log_std.exp())
        return self.distribution.log_prob(x.reshape(-1,*self.input_size)).flatten(1).sum(1)
    
