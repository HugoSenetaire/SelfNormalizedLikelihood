
import torch.nn as nn
import torch
import torch.distributions as distributions
import numpy as np



key_per_sample = ['base_dist_log_prob_samples', 'proposal_log_prob_samples', 'aux_log_prob_samples', 'f_theta_samples', 'b_samples', 'energy']
key_per_batch = ['log_z_estimate', 'z_estimate']

class EBMRegression(nn.Module):
    def __init__(self, energy, proposal, feature_extractor, num_sample_proposal, base_dist = None, explicit_bias = None,  **kwargs):
        super(EBMRegression, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.feature_extractor = feature_extractor
        self.nb_sample = num_sample_proposal
        self.base_dist = base_dist
        self.explicit_bias = explicit_bias
        
        
    def sample_proposal(self, x, nb_sample = 1):
        '''
        Samples from the proposal distribution.
        '''
        if self.feature_extractor is not None:
            x_feature = self.feature_extractor(x)
        else :
            x_feature = x
        samples = self.proposal.sample(x_feature, nb_sample)
        return samples

    def log_prob_proposal(self, x, y):
        '''
        Calculate the log probability of x with the proposal distribution.
        '''
        if self.feature_extractor is not None:
            x_feature = self.feature_extractor(x)
        else :
            x_feature = x
        return self.proposal.log_prob(x_feature, y)
    
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
            out_energy = out_energy + b
            dic_output['b'] = b

        if self.base_dist is not None and use_base_dist :
            if len(x_feature.shape) == 1:
                x_feature = x_feature.unsqueeze(0)
            base_dist_log_prob = self.base_dist.log_prob(x_feature, y).view(x_feature.size(0), -1).sum(1).unsqueeze(1)
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
        

    def estimate_log_z_onepass(self, x_feature, nb_sample = 1000):
        assert nb_sample <= 2048, "The number of samples should be less than 2000 in the one pass"
        dic_output = {}
        
        batch_size = x_feature.shape[0]
        samples = self.proposal.sample(x_feature, nb_sample).to(x_feature.device, x_feature.dtype)
        samples = samples.reshape(x_feature.shape[0]*nb_sample, -1) #(batch_size, num_samples, y_size)
        x_feature_expanded = x_feature.unsqueeze(1).expand(-1, nb_sample, -1).reshape(x_feature.shape[0]*nb_sample, -1) #(batch_size * num_samples, x_size)

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
                aux_log_prob = base_dist_log_prob - samples_log_prob 
                log_z_estimate = torch.logsumexp(-energy_samples + aux_log_prob,dim=1) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device)) 
                z_estimate = log_z_estimate.exp()
                dic_output.update({"base_dist_log_prob_samples" : base_dist_log_prob, "proposal_log_prob_samples" : samples_log_prob, "aux_log_prob_samples" : aux_log_prob})
            else :
                log_z_estimate = (-energy_samples).logsumexp(dim=1) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
                z_estimate = log_z_estimate.exp()
        else :
            samples_log_prob = self.proposal.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2)
            dic_output.update({"proposal_log_prob_samples" : samples_log_prob,})
            log_z_estimate = torch.logsumexp(-energy_samples - samples_log_prob, dim=1) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
            z_estimate = log_z_estimate.exp()
        
        dic_output['log_z_estimate'] = log_z_estimate
        dic_output['z_estimate'] = z_estimate
        
        return log_z_estimate, dic_output
    


    def estimate_log_z(self, x, nb_sample = 1000):
        batch_size = x.shape[0]
        dic_output = {}
        if self.feature_extractor is not None:
            x_feature = self.feature_extractor(x)
        else :
            x_feature = x
        
        if nb_sample <= 2048:
            return self.estimate_log_z_onepass(x_feature, nb_sample = nb_sample)
        else :
            number_iter = np.ceil(nb_sample / 2048)
            last_iter_remaining = nb_sample % 2048
            for k in range(int(number_iter)):
                current_nb_sample = 2048
                if k == number_iter - 1 and last_iter_remaining != 0 :
                    current_nb_sample = last_iter_remaining
                _, current_dic_output = self.estimate_log_z_onepass(x_feature, nb_sample = current_nb_sample)
                for key in current_dic_output :
                    if key in key_per_sample :
                        if key in dic_output.keys() :
                            dic_output[key] +=[current_dic_output[key].clone().detach()]
                        else :
                            dic_output[key] = [current_dic_output[key].clone().detach()]

                    elif key in key_per_batch :
                        if key == 'log_z_estimate' :
                            new_value = current_dic_output[key].clone().detach() + torch.log(torch.tensor(current_nb_sample, dtype=x_feature.dtype, device=x_feature.device))
                            if key in dic_output.keys() :
                                dic_output[key] = torch.logsumexp(torch.cat([new_value.unsqueeze(1), dic_output[key].unsqueeze(1)], dim = 1), dim = 1)
                            else :
                                dic_output[key] = new_value
                        elif key == 'z_estimate' :
                            new_value = current_dic_output[key].clone().detach()*torch.tensor(current_nb_sample, dtype=x_feature.dtype, device=x_feature.device)
                            if key in dic_output.keys() :
                                dic_output[key] += new_value
                            else :
                                dic_output[key] = new_value
                   
        # assert False  
        for key in dic_output.keys() :
            if key == 'log_z_estimate' :
                dic_output[key] = dic_output[key] - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
            elif key == 'z_estimate' :
                dic_output[key] = dic_output[key]/torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device)
            elif key in key_per_sample :
                dic_output[key] = torch.cat(dic_output[key], dim=1)
        return dic_output['log_z_estimate'], dic_output

    
