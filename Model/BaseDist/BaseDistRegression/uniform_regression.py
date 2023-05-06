
import torch.nn as nn
import torch.distributions as distributions
import torch
import numpy as np




class UniformRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, min = 'dataset', max='dataset',  learn_min = False, learn_max = False, **kwargs) -> None:
        super().__init__()
        # assert False
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        
        index = np.random.choice(len(dataset), 10000)
        try :
            data = torch.cat([dataset[i][1] for i in index])
        except RuntimeError:
            data = torch.cat([dataset[i][1].reshape(1, *self.input_size_y) for i in index])

        if min == 'dataset' :
            self.min = data.min(0).values
        else :
            self.min = torch.tensor(min)

        if max == 'dataset' :
            self.max = data.max(0).values
        else :
            self.max = torch.tensor(max)

 
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
    
    