import torch.nn as nn
import torch

class SelfNormalized(nn.Module):
    def __init__(self, energy, proposal, num_sample_proposal, **kwargs):
        super(SelfNormalized, self).__init__()
        self.energy = energy
        # self.log_Z = torch.nn.parameter.Parameter(torch.zeros(1))
        self.proposal = proposal
        self.nb_sample = num_sample_proposal

    def sample(self, nb_sample = 1):
        return self.proposal.sample(nb_sample)
    

    def forward(self, x, nb_sample = None):

        if nb_sample is None:
            nb_sample = self.nb_sample
        # Evaluate energy from the batch
        energy_batch = self.energy(x)
        # Evaluate energy from the samples
        samples = self.proposal.sample(nb_sample).to(x.device, x.dtype)
        # print(samples.to)
        energy_samples = self.energy(samples)
        log_prob_samples = self.proposal.log_prob(samples)
        estimated_z = (-energy_samples-log_prob_samples).exp().mean()

        loss = energy_batch + estimated_z
        likelihood = -loss.mean()

        return loss, {"loss" : loss, "likelihood" : likelihood, "energy_batch" : energy_batch, "z" : estimated_z, "energy_samples" : energy_samples, "log_prob_samples" : log_prob_samples,}

    







    
