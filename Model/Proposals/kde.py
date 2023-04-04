import torch.distributions as dist
import numpy as np
import torch
import sklearn.neighbors

class KernelDensity():
    def __init__(self, input_size, dataset, kernel='gaussian', bandwith = 'scott', **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        
        index = np.random.choice(len(dataset), 10000)
        data = torch.cat([dataset[i][0] for i in index]).numpy()
        self.kd = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwith).fit(data)


    def sample(self, nb_sample = 1):
        samples = torch.from_numpy(self.kd.sample((nb_sample,)))
        return samples
    
    def log_prob(self, x):
        return torch.from_numpy(self.kd.score_samples(x.numpy())).to(x.device, x.dtype)
    
