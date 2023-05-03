import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class StandardGaussianRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size_x = np.prod(input_size_x)
        self.input_size_y = np.prod(input_size_y)
        print("Init Standard Gaussian...")
        index = np.random.choice(len(dataset), 100)
        data = torch.cat([dataset[i][1].reshape(1, self.input_size_y) for i in index])
        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0),)
        else :
            raise NotImplementedError
        
        if std == 'dataset' :
            self.std = nn.parameter.Parameter(data.std(0),)
        else :
            raise NotImplementedError
        
        self.distribution = dist.Normal(self.mean, self.std)
        print("Init Standard Gaussian... end")

    def sample(self, x_feature, nb_sample = 1):
        batch_size = x_feature.shape[0]
        samples = self.distribution.sample((batch_size, nb_sample,)).reshape(batch_size, nb_sample, self.input_size_y).detach()
        return samples
    
    def log_prob(self, x_feature, y):
        assert x_feature.shape[0] == y.shape[0]
        batch_size = x_feature.shape[0]
        return self.distribution.log_prob(y).reshape(batch_size, self.input_size_y).sum(1)
    
