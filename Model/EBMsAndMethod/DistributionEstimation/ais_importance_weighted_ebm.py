import itertools

import torch
import torch.distributions as distributions
import torch.nn as nn
import math 
from .importance_weighted_ebm import ImportanceWeightedEBM
from .differentiable_sampler.differentiable_ula import ULA

from ...Sampler.utils_sampler.clip_sampler import clip_grad, clip_data





class AISImportanceWeightedEBM(ImportanceWeightedEBM):
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
        f_theta,
        proposal,
        base_dist,
        explicit_bias,
        cfg_ebm,
        nb_sample_init_bias=1024,
    ):
        self.nb_transitions_ais = cfg_ebm.nb_transitions_ais
        self.train_ais = cfg_ebm.train_ais
        self.nb_step_ais = cfg_ebm.nb_step_ais
        self.step_size_ais = cfg_ebm.step_size_ais
        self.sigma_ais = cfg_ebm.sigma_ais
        self.clip_max_norm_ais = cfg_ebm.clip_max_norm_ais
        self.clip_max_value_ais = cfg_ebm.clip_max_value_ais
        self.clamp_min_ais = cfg_ebm.clamp_min_ais
        self.clamp_max_ais = cfg_ebm.clamp_max_ais
        super(AISImportanceWeightedEBM, self).__init__(f_theta,
                                                        proposal,
                                                        base_dist,
                                                        explicit_bias,
                                                        cfg_ebm,
                                                        nb_sample_init_bias=nb_sample_init_bias,)



    def get_target(self, k,):
        """
        Get the annealed function to target at iteration k of the AIS algorithm.
        """
        
        value = min(1.0, k / (self.nb_transitions_ais))
        if self.base_dist is None :
            return lambda x: - self.f_theta(x).flatten() * value + self.proposal.log_prob(x).flatten() * (1-value)
        else :
            return lambda x: - self.f_theta(x).flatten() * value + self.base_dist.log_prob(x).flatten()


            
    def sample_ais(self, nb_sample=1, return_log_prob=False, detach_sample = True):
        """
        Samples from the proposal distribution.
        """
        if return_log_prob :
            samples_proposal, samples_proposal_log_prob = self.proposal.sample(nb_sample, return_log_prob=return_log_prob)
            samples_proposal_log_prob = samples_proposal_log_prob.to(next(self.parameters()).device)
        else :
            samples_proposal = self.proposal.sample(nb_sample, return_log_prob = False)
        samples_proposal = samples_proposal.to(next(self.parameters()).device)
        
        if detach_sample:
            samples_proposal = samples_proposal.detach()

    
        samples_proposal.requires_grad_(True)
        dist = torch.distributions.Normal(0, 1)
        for k in range(self.nb_transitions_ais):
            eps = torch.randn_like(samples_proposal)
            target_density = self.get_target(k)
            density_init = target_density(samples_proposal)
            forward_grad = torch.autograd.grad(density_init.sum(), samples_proposal, create_graph=True, only_inputs=True)[0]
            # forward_grad = clip_grad(forward_grad, clip_max_norm=self.clip_max_norm_ais, clip_max_value=self.clip_max_value_ais)
            

            update = math.sqrt(2 * self.step_size_ais) * eps + self.step_size_ais * forward_grad
            new_samples_proposal = samples_proposal + update
            if return_log_prob :
                density_step = target_density(new_samples_proposal)
                backward_grad = torch.autograd.grad(density_step.sum(), new_samples_proposal, create_graph=True, only_inputs=True)[0]
                eps_reverse = (samples_proposal - new_samples_proposal - self.step_size_ais * backward_grad) / math.sqrt(2 * self.step_size_ais)
                samples_proposal_log_prob = samples_proposal_log_prob - dist.log_prob(eps_reverse.flatten(1)).sum(1) + dist.log_prob(eps.flatten(1)).sum(1)
                samples_proposal = new_samples_proposal
            else :
                samples_proposal = new_samples_proposal.detach()
                samples_proposal.requires_grad_(True)


        if return_log_prob :
            return samples_proposal, samples_proposal_log_prob
        else :
            return samples_proposal

    def estimate_log_z(
        self,
        x,
        nb_sample=1000,
        sample_function = None,
        detach_sample=True,
        detach_base_dist=False,
        requires_grad=False,
        return_samples=False,
        noise_annealing=0.0,
        force_calculation=False,
    ):
        if sample_function is None and self.train_ais:
            sample_function = self.sample_ais
        else :
            sample_function = self.sample
        return super().estimate_log_z(
            x,
            nb_sample= nb_sample,
            sample_function=sample_function,
            detach_sample=detach_sample,
            detach_base_dist=detach_base_dist,
            requires_grad=requires_grad,
            return_samples=return_samples,
            noise_annealing=noise_annealing,
            force_calculation=force_calculation,
        )