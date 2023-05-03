
import torch.nn as nn
import torch
import torch.distributions as distributions

class EBMRegression(nn.Module):
    def __init__(self, energy, proposal, feature_extractor, num_sample_proposal, base_dist = None, explicit_bias = None,  **kwargs):
        super(EBMRegression, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.feature_extractor = feature_extractor
        self.nb_sample = num_sample_proposal
        self.base_dist = base_dist
        self.explicit_bias = explicit_bias
        
        
    def sample(self, x, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        return self.proposal.sample(x, nb_sample)
    
    def calculate_energy(self, x, y, use_base_dist = True):
        '''
        Calculate energy of x with the energy function
        '''
        if self.feature_extractor is not None:
            x_feature = self.feature_extractor(x)
        else :
            x_feature = x

        dic_output = {}
        out_energy = self.energy(x_feature, y)
        dic_output['f_theta'] = out_energy

        if self.explicit_bias is not None :
            b = self.explicit_bias(x_feature).reshape(out_energy.shape)
            # print(b)
            out_energy = out_energy + b
            dic_output['b'] = b
        # assert False
        if self.base_dist is not None and use_base_dist :
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            base_dist_log_prob = self.base_dist.log_prob(x, y).view(x.size(0), -1).sum(1).unsqueeze(1)
            # print(base_dist_log_prob.flatten())
            # assert False
            dic_output['base_dist_log_prob'] = base_dist_log_prob
        else :
            base_dist_log_prob = torch.zeros_like(out_energy)
        current_energy = out_energy - base_dist_log_prob
        dic_output['energy'] = current_energy

        return current_energy, dic_output
        
    

    def switch_mode(self, ):
        '''
        Switch the mode of the model and perform renormalization when moving from one mode to another.
        '''
        return None
        

    def estimate_log_z(self, x, nb_sample = 1000):
        batch_size = x.shape[0]
        dic_output = {}
        if self.feature_extractor is not None:
            x_feature = self.feature_extractor(x)
        else :
            x_feature = x
        samples = self.sample(x_feature, nb_sample).to(x.device, x.dtype)
        samples = samples.reshape(x.shape[0]*nb_sample, -1) #(batch_size, num_samples, y_size)
        x_feature_expanded = x_feature.unsqueeze(1).expand(-1, nb_sample, -1).reshape(x.shape[0]*nb_sample, -1) #(batch_size * num_samples, x_size)

        energy_samples = self.energy(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
        dic_output['f_theta_samples'] = energy_samples

        if self.explicit_bias is not None :
            b = self.explicit_bias(x_feature_expanded).reshape(energy_samples.shape)
            energy_samples = energy_samples + b
            dic_output['b_samples'] = b


        if self.base_dist is not None :
            if self.base_dist != self.proposal :
                base_dist_log_prob = self.base_dist.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
                samples_log_prob = self.proposal.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
                aux_prob = base_dist_log_prob - samples_log_prob 
                log_z_estimate = torch.logsumexp(-energy_samples + aux_prob,dim=1) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device)) 
                z_estimate = log_z_estimate.exp()
                dic_output.update({"base_dist_log_prob_samples" : base_dist_log_prob, "proposal_log_prob_samples" : samples_log_prob, "aux_prob_samples" : aux_prob})
            else :
                log_z_estimate = (-energy_samples).logsumexp(dim=1) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))
                z_estimate = log_z_estimate.exp()
        else :
            samples_log_prob = self.proposal.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2)
            dic_output.update({"proposal_log_prob_samples" : samples_log_prob,})
            log_z_estimate = torch.logsumexp(-energy_samples - samples_log_prob, dim=1) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))
            z_estimate = log_z_estimate.exp()
        
        dic_output['log_z_estimate'] = log_z_estimate
        dic_output['z_estimate'] = z_estimate
        
        return log_z_estimate, dic_output
    







    
