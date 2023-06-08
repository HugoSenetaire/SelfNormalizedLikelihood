import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class StandardGaussian(nn.Module):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', feature_extractor = None, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        print("Init Standard Gaussian...")
        index = np.random.choice(len(dataset), min(10000, len(dataset)))
        # data = torch.cat([dataset.__getitem__(i)[0] for i in index]).reshape(-1, *self.input_size)
        with torch.no_grad():
            if feature_extractor is None :
                data = torch.cat([dataset[i][0] for i in index]).reshape(len(index), *input_size)
            else :
                data = torch.cat([feature_extractor(dataset[i][0].unsqueeze(0)) for i in index]).reshape(len(index), -1)
        data += torch.randn_like(data) * 1e-2

        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0), requires_grad=False)
        else :
            raise NotImplementedError

        if std == 'dataset' :
            self.log_std = nn.parameter.Parameter(data.std(0).log(), requires_grad=False)
        else :
            raise NotImplementedError
        
        print("Init Standard Gaussian... end")

    def sample(self, nb_sample = 1):
        # print(self.mean)
        # print(self.log_std)
        self.distribution = dist.Normal(self.mean, self.log_std.exp())
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, *self.input_size).detach()
        return samples
    
    def log_prob(self, x):
        self.distribution = dist.Normal(self.mean, self.log_std.exp())
        print(x.shape)
        print(self.mean.shape)
        print(self.log_std.shape)
        return self.distribution.log_prob(x).flatten(1).sum(1)
    
