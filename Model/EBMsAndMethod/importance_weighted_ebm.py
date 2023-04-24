
import torch.nn as nn
import torch
import torch.distributions as distributions

class ImportanceWeightedEBM(nn.Module):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist = None, learn_base_dist = False, explicit_norm = False, **kwargs):
        super(ImportanceWeightedEBM, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.nb_sample = num_sample_proposal
        self.explicit_norm = explicit_norm

        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc, requires_grad=learn_base_dist)
            self.base_logstd = nn.Parameter(base_dist.scale.log(), requires_grad=learn_base_dist)
            self.base_logweight = nn.Parameter(base_dist.scale.mean() * 0., requires_grad=learn_base_dist)
        else:
            self.base_mu = None
            self.base_logstd = None
        
        if self.explicit_norm:
            self.log_Z = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=False)

            
    def sample(self, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        return self.proposal.sample(nb_sample)
    
    def calculate_energy(self, x):
        '''
        Calculate energy of the samples from the energy function.
        '''
        current_energy = self.energy(x)
        if self.base_mu is not None:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            current_energy += base_dist.log_prob(x).view(x.size(0), -1).sum(1)

        if self.explicit_norm :
            return current_energy - self.log_Z
        else:
            return current_energy

    def estimate_z(self, energy_samples, log_prob_samples):
        raise NotImplementedError
    
    def switch_mode(self, ):
        '''
        Switch the mode of the model and perform renormalization when moving from one mode to another.
        '''
        if self.explicit_norm :
            samples = self.sample(self, nb_sample=1000)
            log_prob_samples = self.proposal.log_prob(samples)
            energy_samples = self.calculate_energy(samples)
            estimated_z = (-energy_samples-log_prob_samples).exp().mean()
            self.log_Z.data = torch.log(estimated_z)
        


    def forward(self, x, nb_sample = None):
        '''
        Forward pass of the model.
        '''
        # Evaluate energy from the batch
        energy_batch = self.calculate_energy(x)


        if nb_sample == 0 :
            loss = energy_batch
            likelihood = -loss.mean()
            return loss, {"loss" : loss, "likelihood" : likelihood, "energy_batch" : energy_batch, "z" : torch.zeros(1), "energy_samples" : torch.zeros(1), "log_prob_samples" : torch.zeros(1),}
        # Evaluate energy from the samples
        if nb_sample is None:
            nb_sample = self.nb_sample

        samples = self.sample(nb_sample).to(x.device, x.dtype)
        energy_samples = self.calculate_energy(samples)

        # Evaluate log prob from the samples
        log_prob_samples = self.proposal.log_prob(samples)

        # Compute the estimated Z
        estimated_z = self.estimate_z(energy_samples, log_prob_samples)

        loss = energy_batch + estimated_z
        likelihood = -loss.mean()

        return loss, {"loss" : loss, "likelihood" : likelihood, "energy_batch" : energy_batch, "z" : estimated_z, "energy_samples" : energy_samples, "log_prob_samples" : log_prob_samples,}

    







    
