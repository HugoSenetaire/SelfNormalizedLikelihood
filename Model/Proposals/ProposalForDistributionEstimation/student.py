import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class StudentProposal(nn.Module):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', feature_extractor = None, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        print("Init Standard Gaussian...")

        # try :

        index = np.random.choice(len(dataset), min(10000, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)[0] for i in index]).reshape(-1, *self.input_size)
        self.mean = nn.parameter.Parameter(data.reshape(-1, *self.input_size).mean(0), requires_grad=True)
        self.log_std = nn.parameter.Parameter(data.reshape(-1, *self.input_size).std(0).log(), requires_grad=True)
        self.log_df = nn.parameter.Parameter(torch.zeros_like(self.mean.data), requires_grad=True)

        print("Init Standard Gaussian... end")

    def sample(self, nb_sample = 1):
        self.distribution = dist.StudentT(self.log_df.exp(), self.mean, self.log_std.exp())
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, *self.input_size).detach()
        return samples
    
    def log_prob(self, x):
        self.distribution = dist.StudentT(self.log_df.exp(), self.mean, self.log_std.exp())
        return self.distribution.log_prob(x.reshape(-1,*self.input_size)).flatten(1).sum(1)
    
