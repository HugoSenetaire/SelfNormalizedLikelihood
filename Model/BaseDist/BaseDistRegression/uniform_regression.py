
import torch.nn as nn
import torch.distributions as distributions
import torch
import numpy as np




class UniformRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, min_data = 'dataset', max_data='dataset',  learn_min = False, learn_max = False, **kwargs) -> None:
        super().__init__()
        # assert False
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        
        if min_data == 'dataset' :
            self.min_data = float('inf') 
            for i in range(len(dataset)):
                self.min_data = min(self.min_data, dataset[i][1])

            self.min_data = torch.tensor(self.min_data -1, dtype=torch.float32)
        else :
            self.min_data = torch.tensor(min_data, dtype=torch.float32)
        if max_data == 'dataset' :
            self.max_data = float('-inf') 
            for i in range(len(dataset)):
                self.max_data = max(self.max_data, dataset[i][1])
            self.max_data= torch.tensor(self.max_data +1, dtype=torch.float32)
        else :
            self.max_data = torch.tensor(max_data, dtype=torch.float32)

 
        self.min = nn.Parameter(self.min, requires_grad=learn_min)
        self.logstd = nn.Parameter(self.max, requires_grad=learn_max)


    def sample(self, x_feature, nb_sample = 1):
        batch_size = x_feature.shape[0]
        distribution = distributions.Uniform(self.min, self.max)
        samples = distribution.rsample((batch_size, nb_sample,)).reshape(batch_size, nb_sample, self.input_size_y).detach()
        return samples
    
    def log_prob(self, x_feature, y):
        assert x_feature.shape[0] == y.shape[0]
        batch_size = x_feature.shape[0]
        y = y.reshape(batch_size, -1)
        distribution = distributions.Uniform(self.min, self.max)
        return distribution.log_prob(y).sum(-1).reshape(batch_size, -1)
    
    