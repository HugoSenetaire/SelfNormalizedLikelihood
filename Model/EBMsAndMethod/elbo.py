import torch.nn as nn
import torch
from .importance_weighted_ebm import ImportanceWeightedEBM

class ELBO(ImportanceWeightedEBM):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist = None, **kwargs):
        super(ELBO, self).__init__(energy=energy, proposal=proposal, num_sample_proposal=num_sample_proposal, base_dist=base_dist, **kwargs)
        self.type_z = "log"


    
