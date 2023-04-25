import torch.nn as nn
import torch
from .importance_weighted_ebm import ImportanceWeightedEBM

class ELBO(ImportanceWeightedEBM):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist = None, **kwargs):
        super(ELBO, self).__init__(energy=energy, proposal=proposal, num_sample_proposal=num_sample_proposal, base_dist=base_dist, **kwargs)

    def estimate_z(self, x, nb_sample):
        samples = self.sample(nb_sample).to(x.device, x.dtype)
        energy_samples = self.calculate_energy(samples, use_base_dist = True)
        log_prob_samples = self.proposal.log_prob(samples)
        estimated_z = torch.logsumexp(-energy_samples-log_prob_samples, dim = 0) - torch.log(torch.tensor(nb_sample, dtype = torch.float32))


        dic_output=  {'log_Z': estimated_z, 'log_prob_samples': log_prob_samples, 'energy_samples': energy_samples}
        return estimated_z, dic_output



    
