import itertools

import torch
import torch.distributions as distributions
import torch.nn as nn
import numpy as np
import tqdm

class ImportanceZEstimator(nn.Module):
    """
    Combine f_theta, bias, proposal and base distribution to form an EBM.

    Attributes :
    ------------
    f_theta : torch.nn.Module
        The neural network function of the EBM.
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
    """

    def __init__(
        self,
        full_energy,
        proposal,
        cfg_ebm,
        nb_sample_init_bias=50000,
    ):
        super(ImportanceZEstimator, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.full_energy = full_energy.to(device)
        self.proposal = proposal.to(device)
        self.nb_sample_init_bias = nb_sample_init_bias
        self.cfg_ebm = cfg_ebm
        self.device=device


        # If we use explicit bias, set it to a first estimation of the normalization constant.
        self.initialize_explicit_bias()

    def initialize_explicit_bias(self,):
        if hasattr(self.explicit_bias, "bias"):
            log_z_estimate, dic = self.multiple_batch_estimate_log_z(torch.zeros(1,dtype=torch.float32,).to(self.device),nb_sample=self.nb_sample_init_bias,)
            self.explicit_bias.bias.data = (log_z_estimate - np.log(self.nb_sample_init_bias)).logsumexp(dim=0).exp().detach()

    def sample(self, nb_sample=1, detach_sample=True, ):
        """
        Samples from the proposal distribution.
        """
        samples, samples_log_prob = self.proposal.sample(nb_sample, return_log_prob=True)
        if detach_sample:
            samples = samples.detach()

        
        return samples, samples_log_prob, {}
            

    
    def multiple_batch_estimate_log_z(self, x, nb_sample=1000, noise_annealing=0.0, force_calculation=False, sample_function=None,):
        """
        Estimate the log normalization constant of the EBM by sampling a large number of element from the proposal sequentially
        and then computing the log normalization constant using the importance weighted estimator.

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
        """
        with torch.no_grad():
            test_sample = self.proposal.sample(1, return_log_prob=False)[0]
            sample_dim = np.prod(test_sample.shape[1:])
            log_sample_dim = np.log2(sample_dim)
            batch_size_max = int(max(32, min(1000, 1000/(2**int(log_sample_dim-6)))))
            # print('Multiple batch estimate log z with batch size max :', batch_size_max,)
            
            nb_batch = int(np.ceil(nb_sample/batch_size_max))
            current_sample_size = min(batch_size_max, nb_sample)
            log_z_estimate, _, _ = self.estimate_log_z(
                                    x,
                                    nb_sample=current_sample_size,
                                    noise_annealing=noise_annealing,
                                    force_calculation=force_calculation,
                                    sample_function=sample_function,
                                    )
            log_z_estimate = (log_z_estimate - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))).logsumexp(dim=0, keepdim=True)
            for i in tqdm.tqdm(range(1,nb_batch), position=2, desc="Multiple batch estimate log z {}".format(batch_size_max), leave=False,):
                nb_sample_current = min(batch_size_max, nb_sample - current_sample_size)
                log_z_estimate_current, _, _ = self.estimate_log_z(x,
                                        nb_sample=nb_sample_current,
                                        noise_annealing=noise_annealing,
                                        force_calculation=force_calculation,
                                        sample_function=sample_function,
                                    )
                log_z_estimate = torch.cat([log_z_estimate + torch.log(torch.tensor(current_sample_size, dtype=x.dtype, device=x.device)), log_z_estimate_current]).logsumexp(dim=0, keepdim=True)
                current_sample_size += nb_sample_current
                log_z_estimate = (log_z_estimate - torch.log(torch.tensor(current_sample_size, dtype=x.dtype, device=x.device))).logsumexp(dim=0, keepdim=True)
        return log_z_estimate, {}

    def estimate_log_z(
        self,
        x,
        nb_sample=1000,
        sample_function = None,
        detach_sample=True,
        requires_grad=False,
        noise_annealing=0.0,
    ):
        """
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

        """
        dic_output = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)


        # Sample from the proposal 
        if sample_function is None:
            sample_function = self.sample
        samples_proposal, samples_proposal_log_prob, dic_sampler = sample_function(nb_sample, detach_sample=detach_sample,)
        dic_output.update({"z_estimation/"+key: value for key, value in dic_sampler.items()})

        samples_proposal = samples_proposal.to(x.device, x.dtype)
        samples_proposal_log_prob = samples_proposal_log_prob.to(x.device, x.dtype).reshape((nb_sample, 1, -1)).sum(1)
        if requires_grad:
            samples_proposal.requires_grad = True


        # Add noise to the samples if noise_annealing is not None :
        if noise_annealing > 1e-4:
            epsilon = torch.rand_like(samples_proposal).to(x.device, x.dtype)
            noise_to_add = epsilon * noise_annealing
            log_prob_noise = torch.distributions.Normal(0, 1).log_prob(epsilon).reshape((samples_proposal.shape[0], -1)).sum(-1)
            samples_proposal_noisy = samples_proposal + noise_to_add
        else:
            log_prob_noise = torch.zeros((nb_sample,1)).to(x.device, x.dtype)
            samples_proposal_noisy = samples_proposal
            

        # Energy :
        energy_proposal, dic_energy = self.full_energy(samples_proposal_noisy)
        dic_output.update({"z_estimation/"+key: value.detach() for key, value in dic_energy.items()})


        # Calculate Log Z :
        log_z_estimate = (energy_proposal - samples_proposal_log_prob -log_prob_noise).flatten()


        # Estimate some feedbacks
        ESS_estimate_num = log_z_estimate.logsumexp(0) * 2
        ESS_estimate_denom = (2 * log_z_estimate).logsumexp(0)
        ESS_estimate = (ESS_estimate_num - ESS_estimate_denom).exp()
        dic_output.update(
            {
                "z_estimation/proposal_loglikelihood_on_sample_proposal": samples_proposal_log_prob.detach(),
                "z_estimation/log_z_estimate": log_z_estimate.detach(),
                "z_estimation/ESS_estimate": ESS_estimate.detach(),
            })
        return log_z_estimate, samples_proposal, dic_output
