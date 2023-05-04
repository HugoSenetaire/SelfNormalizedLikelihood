import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class UniformRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, min='dataset', max ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size_x = np.prod(input_size_x)
        self.input_size_y = np.prod(input_size_y)
        print("Init Standard Gaussian...")
        index = np.random.choice(len(dataset), 100)
        data = torch.cat([dataset[i][1].reshape(1, self.input_size_y) for i in index])
        if min == 'dataset' :
            self.min = data.min(0).values
        else :
            self.min = torch.tensor(min, dtype=torch.float32)
        if max == 'dataset' :
            self.max = data.max(0).values
        else :
            self.max = torch.tensor(max, dtype=torch.float32)

        self.min = nn.parameter.Parameter(self.min,)
        self.max = nn.parameter.Parameter(self.max,)
        
        self.distribution = dist.Uniform(self.min, self.max)
        print("Init Standard Gaussian... end")

    def sample(self, x_feature, nb_sample = 1):
        batch_size = x_feature.shape[0]
        samples = self.distribution.sample((batch_size, nb_sample,)).reshape(batch_size, nb_sample, self.input_size_y).detach()
        return samples
    
    def log_prob(self, x_feature, y):
        assert x_feature.shape[0] == y.shape[0]
        batch_size = x_feature.shape[0]
        return self.distribution.log_prob(y).reshape(batch_size, self.input_size_y).sum(1)
    
