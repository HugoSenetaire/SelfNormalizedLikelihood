import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class StandardGaussian(nn.Module):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        print("Init Standard Gaussian...")
        
        index = np.random.choice(len(dataset), 10000)
        try :
            data = torch.cat([dataset[i][0] for i in index]).reshape(-1, *self.input_size)
        except TypeError :
            data = torch.cat([dataset[i][0][0] for i in index]).reshape(-1, *self.input_size)
        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0), requires_grad=False)
        else :
            raise NotImplementedError
        
        if std == 'dataset' :
            self.std = nn.parameter.Parameter(data.std(0), requires_grad=False)
        else :
            raise NotImplementedError
        
        self.distribution = dist.Normal(self.mean, self.std)
        print("Init Standard Gaussian... end")

    def sample(self, nb_sample = 1):
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, *self.input_size).detach()
        return samples
    
    def log_prob(self, x):
        return self.distribution.log_prob(x).flatten(1).sum(1)
    
