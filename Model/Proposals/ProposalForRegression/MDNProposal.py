
import torch
import numpy as np
from .proposal_regression_utils import MDNProposal_Network
from .abstract_proposal_regression import AbstractProposalRegression

def get_MDNProposalRegression(input_size_x_feature, input_size_y, cfg, ):
    return MDNProposalRegression(input_size_x_feature,
                            input_size_y,
                            cfg.K,
                            )

class MDNProposalRegression(AbstractProposalRegression):
    def __init__(self, input_size_x_feature, input_size_y, K = 4, **kwargs):
        super().__init__(input_size_x_feature=input_size_x_feature, input_size_y=input_size_y)
        self.K = K
        self.input_dim_y = np.prod(input_size_y)
        self.hidden_dim = np.prod(input_size_x_feature)
        self.network = MDNProposal_Network(self.hidden_dim, K, self.input_size_y)
        


    def sample_simple(self, x_feature, nb_sample = 1, ):
        batch_size = x_feature.shape[0]
        means, log_sigma2s, weights = self.network(x_feature.reshape(batch_size, *self.input_size_x_feature))
        sigmas = torch.exp(log_sigma2s/2.0)
        means = means.reshape(-1, self.input_dim_y, self.K) # (shape: (batch_size, self.input_size_y, K))
        sigmas = sigmas.reshape(-1, self.input_dim_y, self.K) # (shape: (batch_size, self.input_size_y, K))

        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        y_samples_K = q_distr.sample(sample_shape=torch.Size([nb_sample])) # (shape: (num_samples, batch_size, self.input_size_y, K))
        
        inds = torch.multinomial(input = weights.flatten(0,1), num_samples=nb_sample, replacement=True).unsqueeze(2).unsqueeze(2) # (shape: (batch_size, num_samples, 1, 1))
        inds = inds.expand(batch_size, nb_sample, self.input_dim_y, 1) # (shape: (batch_size, num_samples, self.input_size_y, 1))
        inds = torch.transpose(inds, 1, 0) # (shape: (num_samples, batch_size, self.input_size_y, 1))
        
        y_samples = y_samples_K.gather(3, inds).squeeze(3) # (shape: (num_samples, batch_size, self.input_size_y))
        y_samples = y_samples.detach()
        y_samples = torch.transpose(y_samples, 1, 0).reshape(batch_size, nb_sample, self.input_dim_y) # (shape: (batch_size, num_samples, input_size_y ))
        return y_samples


    def log_prob_simple(self, x_feature, y):
        batch_size = x_feature.size(0)
        means, log_sigma2s, weights = self.network(x_feature)
        sigmas = torch.exp(log_sigma2s/2.0)

        means = means.reshape(batch_size, self.input_dim_y, self.K) # (shape: (batch_size, self.input_dim_y, K))
        sigmas = sigmas.reshape(batch_size, self.input_dim_y, self.K) # (shape: (batch_size, self.input_dim_y, K))
        weights = weights.reshape(batch_size, self.K) # (shape: (batch_size, K))


        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        y_expanded = y.unsqueeze(-1).expand(batch_size, self.input_dim_y, self.K) # (shape: (batch_size, self.input_dim_y, K))
        log_q_ys_K = q_distr.log_prob(y_expanded).sum(1) # (shape: (batch_size, K)
        log_q_ys = torch.logsumexp(torch.log(weights) + log_q_ys_K, dim=1) # (shape: (batch_size))

        return log_q_ys

