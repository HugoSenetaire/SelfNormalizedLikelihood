import torch
import torch.distributions as distributions
import itertools
import torch.nn as nn


class ExplicitBias(nn.Module):
    '''
    Used to handle the explicit bias of the EBM model.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.explicit_bias = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=True)
    
    def forward(self, x):
        return x + self.explicit_bias


class ImportanceWeightedEBM(nn.Module):
    '''
    Combine f_theta, bias, proposal and base distribution to form an EBM.

    Attributes :
    ------------
    energy : torch.nn.Module
        The energy function of the EBM.
    proposal : torch.nn.Module
        The proposal distribution of the EBM, as implemented in ../Proposal
    base_dist : torch.nn.Module
        The base distribution of the EBM, simply requires a log_prob function.
    explicit_bias : bool
        Whether to use explicit bias or not, if yes, the bias is stored in self.explicit_bias.explicit_bias
    nb_sample_init_bias : int
        The number of samples to use to estimate the explicit bias of the EBM either at the beginning of training.
    
    Methods :
    ---------
    sample(torch.tensor/int : nb_sample = 1) -> torch.tensor (shape : (nb_sample, *self.energy.input_dim))
        Sample from the proposal distribution.
    calculate_energy(torch.tensor : x, bool : use_base_dist = True) -> torch.tensor (shape : (x.shape[0], 1))
        Calculate the energy of x with the energy function.
    estimate_log_z(torch.tensor : x, int : nb_sample = 1000) -> torch.tensor (shape : (1,))
        Estimate the log-normalization of the ebm using the proposal.
    forward(torch.tensor : x) -> torch.tensor (shape : (x.shape[0], 1))
        Forward function of the model giving the full energy, used to pickle the model and multiprocess HMM.
    '''
    def __init__(
        self,
        energy,
        proposal,
        base_dist,
        explicit_bias,
        nb_sample_init_bias=1024,
    ):
        super(ImportanceWeightedEBM, self).__init__()
        self.energy = energy
        self.proposal = proposal
        self.nb_sample_init_bias = nb_sample_init_bias
        self.base_dist = base_dist
        self.explicit_bias = explicit_bias
        
        # If we use explicit bias, set it to a first estimation of the normalization constant.
        if hasattr(self.explicit_bias, 'bias'):
            log_z_estimate, dic = self.estimate_log_z(torch.zeros(1, dtype=torch.float32,), nb_sample = self.nb_sample_init_bias)
            self.explicit_bias.bias.data = log_z_estimate


    def sample(self, nb_sample=1):
        """
        Samples from the proposal distribution.
        """
        return self.proposal.sample(nb_sample)

    def calculate_energy(self, x, use_base_dist=True):
        """
        Calculate energy of x with the energy function.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input to the energy function.
        use_base_dist : bool
            Whether to use the base distribution or not. In some cases, it's not necessary to calculate it (eg, when proposal = base distribution)

        Returns :
        ---------
        current_energy : torch.tensor (shape : (x.shape[0], 1))
            The energy of x.
        dic_output : dict
            A dictionary containing the different components of the energy 
            (ie, f_theta of the data, log-prob of the data under the base distribution,
            explicit bias contribution, complete energy of the data)
        """
        dic_output = {}

        # Get the energy of the samples without base distribution
        f_theta = self.energy(x)
        dic_output["f_theta"] = f_theta
        if (self.training and self.proposal is not None and hasattr(self.proposal,"set_x",)):
            # During training, if the proposal is adapative to the batch, set the x of the proposal to the current batch.
            self.proposal.set_x(x)
        

        # Add the explicit bias contribution to the energy
        f_theta = self.explicit_bias(f_theta)
        dic_output.update({"log_explicit_bias" : self.explicit_bias.bias})

        # Add the base distribution contribution to the energy
        if use_base_dist:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            base_dist_log_prob = (self.base_dist.log_prob(x).view(x.size(0), -1).sum(1).unsqueeze(1))
            dic_output["base_dist_log_prob"] = base_dist_log_prob
        else:
            base_dist_log_prob = torch.zeros_like(f_theta)
        current_energy = f_theta - base_dist_log_prob
        dic_output["energy"] = current_energy

        return current_energy, dic_output


    def estimate_log_z(self, x, nb_sample=1000):
        '''
        Estimate the log-normalization of the ebm using the proposal.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input to the energy function. Not really useful here.
        
        nb_sample : int
            The number of samples to use to estimate the log-normalization.

        Returns :
        ---------
        log_z_estimate : torch.tensor (shape : (1,))
            The log-normalization estimate.
        dic_output : dict
            A dictionary containing the different components of the log-normalization estimate 
            (ie, f_theta of the samples, log-prob of the samples under the base distribution,
            log-prob of the samples under the proposal, auxilary prob of the samples, log-normalization estimate)
        
        '''
        dic_output = {}

        samples_proposal = self.sample(nb_sample).to(x.device, x.dtype)

        # Get the energy of the samples_proposal without base distribution
        f_theta_proposal = self.energy(samples_proposal).view(samples_proposal.size(0), -1).sum(1).unsqueeze(1)
        if self.explicit_bias :
            f_theta_proposal = self.explicit_bias(f_theta_proposal)
        dic_output['f_theta_proposal'] = f_theta_proposal # Store the energy without the base distribution
        
        # Add the base distribution and proposal contributions to the energy if they are different
        if self.base_dist != self.proposal :
            base_dist_log_prob = self.base_dist.log_prob(samples_proposal).view(samples_proposal.size(0), -1).sum(1).unsqueeze(1)
            samples_proposal_log_prob = self.proposal.log_prob(samples_proposal).reshape(samples_proposal.shape[0], -1).sum(1).unsqueeze(1)
            aux_prob = base_dist_log_prob - samples_proposal_log_prob 
            log_z_estimate = (-f_theta_proposal + aux_prob).flatten()

            dic_output.update(
                {"base_dist_log_prob_proposal": base_dist_log_prob,
                "proposal_log_prob_proposal": samples_proposal_log_prob,
                "aux_prob_proposal": aux_prob,
                })
        else:
            log_z_estimate = -f_theta_proposal.flatten() 

            

        ESS_estimate_num = log_z_estimate.logsumexp(0)*2
        ESS_estimate_denom = (2*log_z_estimate).logsumexp(0)
        ESS_estimate = (ESS_estimate_num - ESS_estimate_denom).exp()
        log_z_estimate = torch.logsumexp(log_z_estimate, dim=0) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device)) 

        dic_output["log_z_estimate"] = log_z_estimate
        dic_output["ESS_estimate"] = ESS_estimate
        return log_z_estimate, dic_output

    def forward(self, x):
        """
        Had to create a separate forward function in order to pickle the model and multiprocess HMM.

        Parameters :
        ------------
        x : dict(0:torch.tensor (shape : (x.shape[0], *input_dim)))
            The input to the energy function. Use in a dic with 0 as keys to be compatible with the NUTS sampler.

        Returns :
        ---------
        energy : torch.tensor (shape : (x.shape[0], 1))
            The energy of each x.
        """
        current_x = x[0].to(next(self.energy.parameters()).device, next(self.energy.parameters()).dtype)
        energy, _ = self.calculate_energy(current_x)
        return energy
