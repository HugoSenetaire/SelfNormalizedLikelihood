import torch.distributions as dist
import torch.nn as nn
from .abstract_proposal_regression import AbstractProposalRegression

def get_GaussianRegression(input_size_x_feature, input_size_y, dataset, cfg, ):
    return GaussianRegression(input_size_x_feature,
                            input_size_y,
                            dataset,
                            cfg.mean,
                            cfg.std,
                            )

class GaussianRegression(AbstractProposalRegression):
    def __init__(self, input_size_x_feature, input_size_y, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__(input_size_x_feature=input_size_x_feature, input_size_y=input_size_y)
        print("Init Standard Gaussian...")
        data = self.get_data(dataset, 1000).reshape(-1, *self.input_size_y)
        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0),)
        else :
            raise NotImplementedError
        
        if std == 'dataset' :
            self.std = nn.parameter.Parameter(data.std(0),)
        else :
            raise NotImplementedError
        
        print("Init Standard Gaussian... end")

    def sample_simple(self, x_feature, nb_sample = 1):
        self.distribution = dist.Normal(self.mean, self.std)
        batch_size = x_feature.shape[0]
        samples = self.distribution.sample((batch_size, nb_sample,))
        return samples
    
    def log_prob_simple(self, x_feature, y):
        self.distribution = dist.Normal(self.mean, self.std)
        return self.distribution.log_prob(y)
    
