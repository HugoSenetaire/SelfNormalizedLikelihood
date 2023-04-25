
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
        self.nb_sample_explicit_norm = 1000
        self.base_dist = base_dist
        
        if self.explicit_norm:
            self.log_Z = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=False)

            
    def sample(self, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        return self.proposal.sample(nb_sample)
    
    def calculate_energy(self, x, use_base_dist = True):
        '''
        Calculate energy of the samples from the energy function.
        '''
        current_energy = self.energy(x)
        
        if self.base_dist is not None and use_base_dist :
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            current_energy -= self.base_dist.log_prob(x).view(x.size(0), -1).sum(1).unsqueeze(1)

        if self.explicit_norm :
            return current_energy - self.log_Z
        else:
            return current_energy
        
    

    def estimate_z(self, x, nb_sample):
        raise NotImplementedError
    
    def switch_mode(self, ):
        '''
        Switch the mode of the model and perform renormalization when moving from one mode to another.
        '''
        if self.explicit_norm :
            samples = self.sample(self, nb_sample=self.nb_sample_explicit_norm)
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

        # Compute the estimated Z
        estimated_z, aux_dic = self.estimate_z(x, nb_sample)

        loss = energy_batch + estimated_z
        likelihood = -loss.mean()

        dic_output = {"loss" : loss, "likelihood" : likelihood, "energy_batch" : energy_batch, "z" : estimated_z,}
        dic_output.update(aux_dic)
        return loss, dic_output
        # return loss, {"loss" : loss, "likelihood" : likelihood, "energy_batch" : energy_batch, "z" : estimated_z, "energy_samples" : energy_samples, "log_prob_samples" : log_prob_samples,}

    







    
