
import torch.nn as nn
import torch
import torch.distributions as distributions

class ImportanceWeightedEBM(nn.Module):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist = None,  bias_explicit = False, nb_sample_bias_explicit = 1024, **kwargs):
        super(ImportanceWeightedEBM, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.nb_sample = num_sample_proposal
        self.nb_sample_bias_explicit = nb_sample_bias_explicit
        self.base_dist = base_dist
        self.bias_explicit = bias_explicit
        
        if bias_explicit:
            self.explicit_bias = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=True)
            log_z_estimate, dic = self.estimate_log_z(torch.zeros(1, dtype=torch.float32,), nb_sample = self.nb_sample_bias_explicit)
            self.explicit_bias.data = log_z_estimate
        else :
            self.explicit_bias = None

        

            
    def sample(self, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        return self.proposal.sample(nb_sample)
    
    def calculate_energy(self, x, use_base_dist = True):
        '''
        Calculate energy of x with the energy function
        '''
        dic_output = {}
       
        out_energy = self.energy(x)
        dic_output['f_theta'] = out_energy
        if self.training and self.proposal is not None and hasattr(self.proposal, 'set_x',):
            self.proposal.set_x(x)

        
        if self.base_dist is not None and use_base_dist :
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            base_dist_log_prob = self.base_dist.log_prob(x).view(x.size(0), -1).sum(1).unsqueeze(1)
            dic_output['base_dist_log_prob'] = base_dist_log_prob
        else :
            base_dist_log_prob = torch.zeros_like(out_energy)

        
        current_energy = out_energy - base_dist_log_prob
        dic_output['energy'] = current_energy

        if self.explicit_bias is not None :
            dic_output.update({"log_bias_explicit" : self.explicit_bias})
            return current_energy + self.explicit_bias, dic_output
        else:
            return current_energy, dic_output
        

    def switch_mode(self, ):
        '''
        Switch the mode of the model and perform renormalization when moving from one mode to another.
        '''
        if self.bias_explicit :
            samples = self.sample(self, nb_sample=self.nb_sample_bias_explicit)
            log_prob_samples = self.proposal.log_prob(samples)
            energy_samples = self.calculate_energy(samples)
            estimated_z = (-energy_samples-log_prob_samples).exp().mean()
            self.log_Z.data = torch.log(estimated_z)
        

    def estimate_log_z(self, x, nb_sample = 1000):

        dic_output = {}
        samples = self.sample(nb_sample).to(x.device, x.dtype)
        

        energy_samples = self.energy(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
        if self.bias_explicit :
            energy_samples = energy_samples + self.explicit_bias
        dic_output['f_theta_samples'] = energy_samples
        
        if self.base_dist is not None :
            base_dist_log_prob = self.base_dist.log_prob(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
            if self.base_dist != self.proposal :
                base_dist_log_prob = self.base_dist.log_prob(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
                samples_log_prob = self.proposal.log_prob(samples).reshape(samples.shape[0], -1).sum(1).unsqueeze(1)
                aux_prob = base_dist_log_prob - samples_log_prob 
                log_z_estimate = torch.logsumexp(-energy_samples + aux_prob,dim=0) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device)) 
                z_estimate = log_z_estimate.exp()
                dic_output.update({"base_dist_log_prob_samples" : base_dist_log_prob, "proposal_log_prob_samples" : samples_log_prob, "aux_prob_samples" : aux_prob})
            else :
                log_z_estimate = (-energy_samples.flatten()).logsumexp(dim=0) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))
                z_estimate = log_z_estimate.exp()
        else :
            samples_log_prob = self.proposal.log_prob(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
            dic_output.update({"proposal_log_prob_samples" : samples_log_prob, "aux_prob_samples" : aux_prob})
            log_z_estimate = torch.logsumexp(-energy_samples - samples_log_prob, dim=0) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))
            z_estimate = log_z_estimate.exp()
        

        dic_output['log_z_estimate'] = log_z_estimate
        
        dic_output['z_estimate'] = z_estimate

        
        return log_z_estimate, dic_output
    
    def forward(self, x):
        """
        Hade to create a separate forward function in order to pickle the model and multiprocess HMM.
        """
        current_x = x[0].to(next(self.energy.parameters()).device, next(self.energy.parameters()).dtype)
        energy, _ = self.calculate_energy(current_x)
        return energy