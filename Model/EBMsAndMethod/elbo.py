import torch.nn as nn
import torch
from .importance_weighted_ebm import ImportanceWeightedEBM

class ELBO(ImportanceWeightedEBM):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist = None, **kwargs):
        super(ELBO, self).__init__(energy=energy, proposal=proposal, num_sample_proposal=num_sample_proposal, base_dist=base_dist, **kwargs)

    def estimate_z(self, energy_samples, log_prob_samples):
        nb_sample = energy_samples.shape[0]
        estimated_z = torch.logsumexp(-energy_samples-log_prob_samples, dim = 0) - torch.log(torch.tensor(nb_sample, dtype = torch.float32))
        return estimated_z
    




    
