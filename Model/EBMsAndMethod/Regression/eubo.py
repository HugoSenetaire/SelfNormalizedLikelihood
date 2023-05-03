
import torch.nn as nn
import torch
import torch.distributions as distributions
from .global_ebm import EBMRegression

class EUBORegression(nn.Module):
    def __init__(self, energy, proposal, feature_extractor, num_sample_proposal, base_dist = None,  **kwargs):
        super(EUBORegression, self).__init__(energy, proposal, feature_extractor, num_sample_proposal, base_dist, **kwargs)
        self.type_z = 'log'

        