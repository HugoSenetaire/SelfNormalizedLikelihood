
import torch.nn as nn
import torch
import torch.distributions as distributions
import numpy as np

from ...Energy.FeatureExtractor.default_feature_extractor import MockFeatureExtractor
from ...Energy.ExplicitBiasForRegression.mock_explicit_bias import MockBiasRegression



class MockBaseDist(nn.Module):
    '''
    Mock base distribution returning 0. 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def log_prob(self, x, y):
        '''
        Mock log probability returning 0.
        '''
        return torch.zeros(y.shape[0], 1, dtype=x.dtype, device=x.device)


class EBMRegression(nn.Module):
    '''
    Combine f_theta, proposal, base distribution and explicit bias to form an EBM and calculating its log-normalization.

    Attributes :
    ------------
    energy : torch.nn.Module
        The energy function of the EBM.

    proposal : torch.nn.Module
        The proposal distribution of the EBM, as implemented in ../Proposal/ProposalForRegression
        Should implement sample(x_feature) and log_prob(x_feature,y) functions.

    feature_extractor : torch.nn.Module
        The feature extractor of the EBM, as implemented in ../FeatureExtractor/FeatureExtractorForRegression
        If None, will use a MockFeatureExtractor returning the input.
    
    num_sample_proposal : int
        The number of samples to sample from to evaluate log_z.

    base_dist : torch.nn.Module
        The base distribution of the EBM, simply requires a log_prob(x_feature,y) function.
        If None, will use a MockBaseDist returning 0.
    
    explicit_bias : torch.nn.Module
        The explicit b_{\phi} of the EBM, should approximate the normalization constant of the EBM for each x_feature.

    Methods :
    ---------
    sample_proposal(torch.tensor : x , int : nb_sample = 1) -> torch.tensor (shape : (x.shape[0], nb_sample, input_dim))
        Sample from the proposal distribution for each given input x. Nb sample for each input.
    log_prob_proposal(torch.tensor : x, torch.tensor : y) -> torch.tensor (shape : (x.shape[0], 1))
        Calculate the log probability of x,y with the proposal distribution. X and Y should have the same first dimension.
    calculate_energy(torch.tensor : x, torch.tensor : y, bool : use_base_dist = True) -> torch.tensor (shape : (x.shape[0], 1))
        Calculate the energy of x,y with the energy function. x and y should have the same first dimension.
    estimate_log_z_onepass(torch.tensor : x, int : nb_sample = 1000) -> torch.tensor (shape : (x.shape[0], 1))
        Estimate the log-normalization of the ebm using the proposal in one pass. 
        If the number of sample is too high, it will raise an error, use estimate_log_z instead.
    
    '''


    def __init__(self, energy, proposal, feature_extractor, base_dist = None, explicit_bias = None,  **kwargs):
        super(EBMRegression, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.feature_extractor = feature_extractor

        if base_dist is None :
            base_dist = MockBaseDist()
        self.base_dist = base_dist

        if explicit_bias is None :
            explicit_bias = MockBiasRegression(input_size = energy.input_size_x_feature)
        self.explicit_bias = explicit_bias
        
        
    def sample_proposal(self, x, nb_sample = 1, x_feature = None):
        '''
        Samples from the proposal distribution.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_size_x))
            The input to sample from.
        nb_sample : int
            The number of samples per input.
        x_feature : torch.tensor (shape : (x.shape[0], *input_size_x))
            The feature of the input, if None, will be calculated using self.feature_extractor(x).
        
        Returns :
        ---------
        samples : torch.tensor (shape : (x.shape[0], nb_sample, *input_size_x)) 
            The samples from the proposal distribution.
        '''
        if x_feature is None :
            x_feature = self.feature_extractor(x)
        return self.proposal.sample(x_feature, nb_sample)

    def log_prob_proposal(self, x, y, x_feature = None):
        '''
        Calculate the log probability of x with the proposal distribution.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input for regression.
        y : torch.tensor (shape : (x.shape[0], *input_dim))
            The output for regression.
        x_feature : torch.tensor (shape : (x.shape[0], *input_size_x))
            The feature of the input, if None, will be calculated using self.feature_extractor(x).
        '''
        if x_feature is None :
            x_feature = self.feature_extractor(x)
        return self.proposal.log_prob(x_feature, y)
    
    def calculate_energy(self, x, y, use_base_dist = True, x_feature = None):
        '''
        Calculate energy of x with the energy function.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input for regression.
        y : torch.tensor (shape : (x.shape[0], *input_dim))
            The output for regression.
        use_base_dist : bool
            Whether to use the base distribution or not. Might not be necessary if the base distribution is the same as the proposal.
        x_feature : torch.tensor (shape : (x.shape[0], *input_size_x))
            The feature of the input, if None, will be calculated using self.feature_extractor(x).

        Returns :
        ---------
        current_energy : torch.tensor (shape : (x.shape[0], 1))
            The complete energy of x.
        dic_output : dict
            A dictionary containing the different components of the energy (f_theta, b_{\phi}, base_dist_log_prob, energy).
        '''
        dic_output = {}
        if x_feature is None :
            x_feature = self.feature_extractor(x)

        f_theta = self.energy(x_feature, y)
        dic_output['f_theta'] = f_theta

        b = self.explicit_bias(x_feature).reshape(f_theta.shape)
        energy_data = f_theta + b
        dic_output['b'] = b

        if use_base_dist :
            if len(x_feature.shape) == 1:
                x_feature = x_feature.unsqueeze(0)
            base_dist_log_prob = self.base_dist.log_prob(x_feature, y).view(x_feature.size(0), -1).sum(1).unsqueeze(1)
            dic_output['base_dist_log_prob'] = base_dist_log_prob
        else :
            base_dist_log_prob = torch.zeros_like(energy_data)
        
        current_energy = energy_data - base_dist_log_prob
        dic_output['energy'] = current_energy

        return current_energy, dic_output
        

    def _estimate_log_z_onepass(self, x, nb_sample = 1000, x_feature = None):
        '''
        Estimate the log-normalization of the ebm using the proposal in one pass. If number of sample is too high, will raise an error.
        Use estimate_log_z instead.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input for regression.
        nb_sample : int
            The number of samples to use to estimate the log-normalization.
        x_feature : torch.tensor (shape : (x.shape[0], *input_size_x))
            The feature of the input, if None, will be calculated using self.feature_extractor(x).

        Returns :
        ---------
        log_z_estimate : torch.tensor (shape : (x.shape[0], 1))
            The log-normalization estimate.

        dic_output : dict
            A dictionary containing the different components of the energy (f_theta_proposal, b_{\phi}, base_dist_log_prob, energy).
        '''
        assert nb_sample <= 2048, "The number of samples should be less than 2048 in the one pass"
        dic_output = {}
        if x_feature is None :
            x_feature = self.feature_extractor(x)
        batch_size = x_feature.shape[0]
        
        # Get samples and associated features :
        samples = self.proposal.sample(x_feature, nb_sample).to(x_feature.device, x_feature.dtype)
        samples = samples.reshape(x_feature.shape[0]*nb_sample, -1) #(batch_size, num_samples, y_size)
        x_feature_expanded = x_feature.reshape(batch_size, -1).unsqueeze(1).expand(-1, nb_sample, -1).reshape(x_feature.shape[0]*nb_sample, -1) #(batch_size * num_samples, x_size)

        # Calculate f_theta for each sample :
        f_theta_proposal = self.energy(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
        dic_output['f_theta_proposal'] = f_theta_proposal

        b = self.explicit_bias(x_feature_expanded).reshape(f_theta_proposal.shape)
        energy_proposal = f_theta_proposal + b
        dic_output['b_proposal'] = b

        # Depending if the base distribution is the same as the proposal or not, we calculate the log_z estimate differently:
        if self.base_dist != self.proposal :
            base_dist_log_prob = self.base_dist.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
            samples_log_prob = self.proposal.log_prob(x_feature_expanded, samples).view(batch_size, nb_sample, -1).sum(2) #(batch_size, num_samples)
            aux_log_prob = base_dist_log_prob - samples_log_prob 
            log_z_estimate = torch.logsumexp(-energy_proposal + aux_log_prob,dim=1) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device)) 
            dic_output.update({"base_dist_log_prob_samples" : base_dist_log_prob, "proposal_log_prob_samples" : samples_log_prob, "aux_log_prob_samples" : aux_log_prob})
        else :
            log_z_estimate = (-energy_proposal).logsumexp(dim=1) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
        
        dic_output['log_z_estimate'] = log_z_estimate

        return log_z_estimate, dic_output
    


    def estimate_log_z(self, x, nb_sample = 1000, x_feature = None):
        '''
        Estimate the log-normalization of the ebm using the proposal. If the number of sample is too large, we repeat the operation multiple times.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input for regression.
        nb_sample : int
            The number of samples to use to estimate the log-normalization. The limit is fixed at 2048 but should be something changing depending on available memory.
        x_feature : torch.tensor (shape : (x.shape[0], *input_size_x))
            The feature of the input, if None, will be calculated using self.feature_extractor(x).

        Returns :
        ---------
        log_z_estimate : torch.tensor (shape : (x.shape[0], 1))
            The log-normalization estimate.
        
        dic_output : dict
            A dictionary containing the different components of the energy (f_theta_proposal, b_{\phi}, base_dist_log_prob, energy).
        '''
        dic_output = {}
        if x_feature is None :
            x_feature = self.feature_extractor(x)
        
        if nb_sample <= 2048: # If the number of sample is small enough, we can use the one pass method.
            return self._estimate_log_z_onepass(x = x, nb_sample = nb_sample, x_feature=x_feature)
        else :
            number_iter = np.ceil(nb_sample / 2048)
            last_iter_remaining = nb_sample % 2048
            for k in range(int(number_iter)):
                # Repeat the estimate log z one pass method for each iteration :
                current_nb_sample = 2048
                if k == number_iter - 1 and last_iter_remaining != 0 :
                    current_nb_sample = last_iter_remaining
                _, current_dic_output = self.estimate_log_z_onepass(x_feature, nb_sample = current_nb_sample)

                # Update the dic_output :
                for key in current_dic_output :
                    if key == 'log_z_estimate' :
                        new_value = current_dic_output[key].clone().detach() + torch.log(torch.tensor(current_nb_sample, dtype=x_feature.dtype, device=x_feature.device))
                        if key in dic_output.keys() :
                            dic_output[key] = torch.logsumexp(torch.cat([new_value.unsqueeze(1), dic_output[key].unsqueeze(1)], dim = 1), dim = 1)
                        else :
                            dic_output[key] = new_value
                    else :
                        if key in dic_output.keys() :
                            dic_output[key] +=[current_dic_output[key].clone().detach()]
                        else :
                            dic_output[key] = [current_dic_output[key].clone().detach()]

        for key in dic_output.keys() :
            if key == 'log_z_estimate' :
                dic_output[key] = dic_output[key] - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
            else :
                dic_output[key] = torch.cat(dic_output[key], dim=1)

        return dic_output['log_z_estimate'], dic_output

    
