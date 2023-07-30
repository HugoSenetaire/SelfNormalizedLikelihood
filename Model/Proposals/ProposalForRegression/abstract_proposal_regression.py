import torch.nn as nn
import numpy as np
import torch

class AbstractProposalRegression(nn.Module):
    '''
    Abstract class for proposal regression. 
    When implementing a new proposal, you should inherit from this class and implement the following methods:
    - log_prob_simple : Compute the log probability of the proposal.
    - sample_simple : Sample from the proposal.
    '''
    def __init__(self, input_size_x_feature, input_size_y,):
        super().__init__()
        self.input_size_x_feature = input_size_x_feature
        self.input_size_y = input_size_y

    
    def get_data(self, dataset, nb_sample_for_init):
        '''
        Consider a subset of data for initialization
        '''
        index = np.random.choice(len(dataset), min(10000, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)['target'] for i in index]).reshape(-1, *self.input_size_y)
        return data

    def log_prob_simple(self, x_feature, y):
        raise NotImplementedError
    
    def sample_simple(self, x_feature, nb_sample):
        raise NotImplementedError

    def log_prob(self, x_feature, y):
        '''
        Compute the log probability of the proposal.
        '''
        assert x_feature.shape[0] == y.shape[0]
        log_prob = self.log_prob_simple(x_feature, y)
        return log_prob
    
    def sample(self, x_feature, nb_sample):
        '''
        Sample from the proposal.
        '''
        samples = self.sample_simple(x_feature, nb_sample).reshape(x_feature.shape[0], nb_sample, *self.input_size_y)
        return samples