import torch.distributions as dist
import numpy as np
import torch

class StandardGaussian():
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        
        index = np.random.choice(len(dataset), 10000)
        data = torch.cat([dataset[i][0] for i in index])
        if mean == 'dataset' :
            self.mean = data.mean(0)
        else :
            raise NotImplementedError
        
        if std == 'dataset' :
            self.std = data.std(0)
        else :
            raise NotImplementedError
        
        self.distribution = dist.Normal(self.mean, self.std)

    def sample(self, nb_sample = 1):
        samples = self.distribution.sample((nb_sample,))
        return samples
    
    def log_prob(self, x):
        return self.distribution.log_prob(x).flatten(1).sum(1)
    
