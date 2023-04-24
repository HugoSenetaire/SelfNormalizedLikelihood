import torch.nn as nn
import torch
from .importance_weighted_ebm import ImportanceWeightedEBM

class SelfNormalized(ImportanceWeightedEBM):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist=None, switch_mode = None, **kwargs):
        super(SelfNormalized, self).__init__(energy=energy, proposal=proposal, num_sample_proposal=num_sample_proposal, base_dist=base_dist, **kwargs)
        if switch_mode is None or switch_mode<=0 :
            self.type_z = "self_normalized"
        else :
            self.type_z = "elbo"
            
    
    def estimate_z(self, energy_samples, log_prob_samples):
        nb_sample = energy_samples.shape[0]
        if self.type_z == "elbo":
            estimated_z = torch.logsumexp(-energy_samples-log_prob_samples, dim = 0) - torch.log(torch.tensor(nb_sample, dtype = torch.float32))
        elif self.type_z == 'self_normalized':
            estimated_z = (-energy_samples-log_prob_samples).exp().mean()
        return estimated_z
    

    def switch_mode(self, ):
        """
        When changing mode, we need to re-estimate the partition function.
        """
        super(SelfNormalized, self).switch_mode()
        self.type_z = "self_normalized"




    
