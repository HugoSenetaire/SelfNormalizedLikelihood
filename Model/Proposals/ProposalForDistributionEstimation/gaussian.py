import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn
from .abstract_proposal import AbstractProposal


def get_Gaussian(input_size, dataset, cfg, ):
    return Gaussian(input_size,
                            dataset,
                            cfg.mean,
                            cfg.std,
                            cfg.nb_sample_estimate,
                            )

class Gaussian(AbstractProposal):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', nb_sample_estimate = 10000, **kwargs) -> None:
        super().__init__(input_size=input_size)
        print("Init Standard Gaussian...")
        data = self.get_data(dataset, nb_sample_estimate)

        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0), requires_grad=False)
        else :
            raise NotImplementedError

        if std == 'dataset' :
            self.log_std = nn.parameter.Parameter(data.std(0).log(), requires_grad=False)
        else :
            raise NotImplementedError
        
        print("Init Standard Gaussian... end")

    def sample_simple(self, nb_sample = 1):
        self.distribution = dist.Normal(self.mean, self.log_std.exp())
        samples = self.distribution.sample((nb_sample,))
        return samples
    
    def log_prob_simple(self, x):
        self.distribution = dist.Normal(self.mean, self.log_std.exp())
        return self.distribution.log_prob(x).flatten(1).sum(1)
    