import torch.nn as nn


class FullEBM(nn.module):
    """
    Module that concatenates the base distribution, the energy and the bias.
    """
    def __init__(self, base_dist, fnn, bias):
            super().__init__()
            self.base_dist = base_dist
            self.fnn = fnn
            self.bias = bias
    
    def forward(self, x):
        f_theta = self.fnn(x) 
        base_dist_log_prob = self.base_dist.log_prob(x) 
        bias = self.bias(x)
        energy = f_theta - base_dist_log_prob + bias
        dic = {
             "f_theta": f_theta,
             "base_dist_log_prob": base_dist_log_prob,
             "bias": bias,
             "energy": energy,
             }
        return energy, dic

