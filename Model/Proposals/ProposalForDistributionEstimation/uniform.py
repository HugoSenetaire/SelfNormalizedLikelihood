import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from .abstract_proposal import AbstractProposal


def get_Uniform(
    input_size,
    dataset,
    cfg,
):
    return Uniform(
        input_size,
        dataset,
        min_uniform = cfg.min_data,
        max_uniform =cfg.max_data,
        nb_sample_estimate = cfg.nb_sample_estimate,
        shift_min = cfg.shift_min,
        shift_max = cfg.shift_max,
    )


class Uniform(AbstractProposal):
    def __init__(
        self,
        input_size,
        dataset,
        min_uniform="dataset",
        max_uniform="dataset",
        nb_sample_estimate=10000,
        shift_min=0,
        shift_max=0,
        **kwargs
    ) -> None:
        super().__init__(input_size=input_size)
        print("Init Uniform...")
        data = self.get_data(dataset, nb_sample_estimate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        data = data.to(self.device)


        # print(data.min(0)[0].shape)
        if isinstance(min_uniform, float):
            self.min_data = nn.parameter.Parameter(torch.full_like(data[0], fill_value=min_uniform), requires_grad=False)
        else :
            self.min_data= nn.parameter.Parameter(data.min(0)[0]+torch.tensor(shift_min, device=data.device), requires_grad=False)
       
    
        if isinstance(max_uniform, float):
            self.max_data = nn.parameter.Parameter(torch.full_like(data[0], fill_value=max_uniform), requires_grad=False)
        else :
            self.max_data= nn.parameter.Parameter(data.min(0)[0]+torch.tensor(shift_max, device=data.device), requires_grad=False)

        print(self.min_data, self.max_data)

        print("Init Standard Gaussian... end")

    def sample_simple(self, nb_sample=1):
        self.distribution = dist.Uniform(self.min_data, self.max_data)
        samples = self.distribution.sample((nb_sample,))
        return samples

    def log_prob_simple(self, x):
        self.distribution = dist.Uniform(self.min_data, self.max_data)
        x = x.to(self.device)
        return self.distribution.log_prob(x).flatten(1).sum(1)
