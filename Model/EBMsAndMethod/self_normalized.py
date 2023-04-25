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
            
    
    def estimate_z(self, x, nb_sample):
        samples = self.sample(nb_sample).to(x.device, x.dtype)

        energy_samples = self.calculate_energy(samples, use_base_dist=(self.proposal != self.base_dist))

        if self.type_z == "elbo":
            log_prob_samples = self.proposal.log_prob(samples)
            estimated_z = torch.logsumexp(-energy_samples-log_prob_samples, dim = 0) - torch.log(torch.tensor(nb_sample, dtype = torch.float32))
        elif self.type_z == 'self_normalized':
            if self.base_dist == self.proposal:
                log_prob_samples = None
                estimated_z = (-energy_samples).exp().mean()
            else :
                log_prob_samples = self.proposal.log_prob(samples)
                estimated_z = (-energy_samples-log_prob_samples).exp().mean()
        
        dic_output = {"energy_samples" : energy_samples, }
        if log_prob_samples is not None :
            dic_output["log_prob_samples"] = log_prob_samples

    
        return estimated_z, dic_output



    def switch_mode(self, ):
        """
        When changing mode, we need to re-estimate the partition function.
        """
        super(SelfNormalized, self).switch_mode()
        self.type_z = "self_normalized"




    
