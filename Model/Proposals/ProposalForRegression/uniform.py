import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class UniformRegression(nn.Module):
    def __init__(self, input_size_x, input_size_y, dataset, min_data='dataset', max_data ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size_x = np.prod(input_size_x)
        self.input_size_y = np.prod(input_size_y)
        print("Init UNIFORM...")
        if not isinstance(dataset, list):
            dataset = [dataset]
        if min_data == 'dataset' :
            self.min_data = dataset[0][0][1]
            for current_dataset in dataset :
                for i in range(len(current_dataset)):
                    self.min_data = torch.min(self.min_data, current_dataset[i][1])

            self.min_data = torch.tensor(self.min_data -3, dtype=torch.float32)
        else :
            self.min_data = torch.tensor(min_data, dtype=torch.float32)
        if max_data == 'dataset' :
            self.max_data = dataset[0][0][1]
            for current_dataset in dataset :
                for i in range(len(current_dataset)):
                    self.max_data = torch.max(self.max_data, current_dataset[i][1])
            self.max_data= torch.tensor(self.max_data +3, dtype=torch.float32)
        else :
            self.max_data = torch.tensor(max_data, dtype=torch.float32)

        self.min_data = nn.parameter.Parameter(self.min_data,)
        self.max_data = nn.parameter.Parameter(self.max_data,)


        print("Minimum data for uniform proposal : ", self.min_data)
        print("Maximum data for uniform proposal : ", self.max_data)
        self.distribution = dist.Uniform(self.min_data, self.max_data)
        print("Init Standard Gaussian... end")

    def sample(self, x_feature, nb_sample = 1):
        batch_size = x_feature.shape[0]
        samples = self.distribution.sample((batch_size, nb_sample,)).reshape(batch_size, nb_sample, self.input_size_y).detach()
        return samples
    
    def log_prob(self, x_feature, y):
        assert x_feature.shape[0] == y.shape[0]
        batch_size = x_feature.shape[0]
        y_clamp = y.clamp(self.min_data, self.max_data)
        log_prob = self.distribution.log_prob(y_clamp).reshape(batch_size, self.input_size_y).sum(1)
        return log_prob
    
