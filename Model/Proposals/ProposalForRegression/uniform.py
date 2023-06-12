import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn
from .abstract_proposal_regression import AbstractProposalRegression

class UniformRegression(AbstractProposalRegression):
    def __init__(self, input_size_x_feature, input_size_y, dataset, min_data='dataset', max_data ='dataset', shift_min = 0, shift_max = 0, **kwargs) -> None:
        super().__init__(input_size_x_feature=input_size_x_feature, input_size_y=input_size_y)
        print("Init UNIFORM...")

        if min_data == 'dataset' :
            self.min_data = dataset[0][0][1]
            for current_dataset in dataset :
                for i in range(len(current_dataset)):
                    self.min_data = torch.min(self.min_data, current_dataset[i][1])
            self.min_data = torch.tensor(self.min_data, dtype=torch.float32)
            self.min_data = self.min_data - shift_min
        else :
            self.min_data = torch.tensor(min_data, dtype=torch.float32)

        
        if max_data == 'dataset' :
            self.max_data = dataset[0][0][1]
            for current_dataset in dataset :
                for i in range(len(current_dataset)):
                    self.max_data = torch.max(self.max_data, current_dataset[i][1])
            self.max_data= torch.tensor(self.max_data, dtype=torch.float32)
            self.max_data = self.max_data + shift_max
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
        if torch.any(y < self.min_data) or torch.any(y > self.max_data) :
            raise ValueError("The value is outside the range of the uniform distribution, augment the shift")
        batch_size = x_feature.shape[0]
        log_prob = self.distribution.log_prob(y).reshape(batch_size, *self.input_size_y).sum(1)
        return log_prob
    
