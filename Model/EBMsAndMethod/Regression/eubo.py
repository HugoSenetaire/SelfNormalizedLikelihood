
import torch.nn as nn
import torch
import torch.distributions as distributions
from .global_ebm import EBMRegression

class EUBORegression(EBMRegression):
    def __init__(self, energy, proposal, feature_extractor, num_sample_proposal, base_dist = None, explicit_bias = None, **kwargs):
        super(EUBORegression, self).__init__(energy, proposal, feature_extractor, num_sample_proposal, base_dist, explicit_bias = explicit_bias, **kwargs)
        self.type_z = 'log'

        