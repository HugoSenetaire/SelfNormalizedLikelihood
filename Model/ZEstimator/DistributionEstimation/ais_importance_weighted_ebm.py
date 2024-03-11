import itertools
import math

import torch
import torch.distributions as distributions
import torch.nn as nn
import numpy as np

from ...Sampler.utils_sampler.clip_sampler import clip_data, clip_grad
from .importance_weighted_ebm import ImportanceZEstimator


class AISZEstimator(ImportanceZEstimator):
    """
    Combine Energy and Proposal to estimate the log-normalization of the EBM using Annealed Importance Sampling.

    Attributes :
    ------------
    Energy : torch.nn.Module
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
    """

    def __init__(
        self,
        energy,
        proposal,
        cfg_ebm,
        nb_sample_init_bias=50000,
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

        self.adaptive_step_size = cfg_ebm.adaptive_step_size
        self.acceptance_rate_target = cfg_ebm.acceptance_rate_target
        self.variance_sensitive_step = cfg_ebm.variance_sensitive_step
        self.alpha_stepsize_increase = cfg_ebm.alpha_stepsize_increase
        self.alpha_stepsize_decrease =  cfg_ebm.alpha_stepsize_decrease
        self.step_size_min = cfg_ebm.step_size_min
        self.step_size_max = cfg_ebm.step_size_max

        self.step_size = [torch.tensor(self.step_size_ais) for _ in range(self.nb_transitions_ais)]
        if self.variance_sensitive_step:
            self.gamma_0 = [torch.tensor(0.1) for _ in range(self.nb_transitions_ais)] 
        

        super().__init__(
            energy=energy,
            proposal=proposal,
            cfg_ebm=cfg_ebm,
            nb_sample_init_bias=nb_sample_init_bias,
            )


        


    def get_target(self, k,):
        """
        Get the annealed function to target at iteration k of the AIS algorithm.
        """
 
        value = min(1.0, k / (self.nb_transitions_ais))
        assert value > 0 and value <= self.nb_transitions_ais
        return lambda x: -self.energy(x)[0].flatten() * value + self.proposal.log_prob(x).flatten() * (1-value)

    def update_stepsize(self, accept_rate=None, current_tran_id=None, current_gradient_batch=None):
        '''
        Stepsize update machinery.
        :param accept_rate: List of mean acceptance rates after each transitions.
        :param current_tran_id: Current transition id
        :param current_gradient_batch: Current batch of gradients of target logdensity wrt inputs
        :return:
        '''
        dic = {}
        if self.training and self.adaptive_step_size:
            if not self.variance_sensitive_step:
                # for l in range(0, self.nb_transitions_ais):
                if accept_rate[current_tran_id].mean() < self.acceptance_rate_target:
                    
                    self.step_size[current_tran_id] = self.alpha_stepsize_decrease * self.step_size[current_tran_id]
                else:
                    self.step_size[current_tran_id] = self.alpha_stepsize_increase * self.step_size[current_tran_id]

                self.step_size[current_tran_id] = self.step_size[current_tran_id].clamp(self.step_size_min, self.step_size_max)
                dic[f'ais/step_size_{current_tran_id}'] = self.step_size[current_tran_id].detach().cpu()
            else:
                with torch.no_grad():
                    gradient_std = torch.std(current_gradient_batch, dim=0).flatten()
                    if gradient_std.shape[0]<10 :
                        for i in range(gradient_std.shape[0]):
                            dic['ais/gradient_std_'+str(i)] = gradient_std[i].detach().cpu()
                    else :
                        dic['ais/gradient_std_mean'] = gradient_std.mean().detach().cpu()
                        dic['ais/gradient_std_max'] = gradient_std.max()[0].detach().cpu()
                        dic['ais/gradient_std_min'] = gradient_std.min()[0].detach().cpu()
                    self.step_size[current_tran_id] = 0.9 * self.step_size[current_tran_id] + 0.1 * self.gamma_0[
                        current_tran_id] / (gradient_std + 1.)

                    if accept_rate[current_tran_id].mean() < self.acceptance_rate_target:
                        self.gamma_0[current_tran_id] *= 0.99
                    else:
                        self.gamma_0[current_tran_id] *= 1.02
                    
                    dic[f'ais/gamma_0_{current_tran_id}'] = self.gamma_0[current_tran_id].detach().cpu()
                    dic[f'ais/acceptrate_{current_tran_id}'] = accept_rate[current_tran_id].mean().detach().cpu()

                    self.step_size[current_tran_id] = self.step_size[current_tran_id].clamp(self.step_size_min, self.step_size_max)
                    # if self.step_size
                    dic[f'ais/step_size_{current_tran_id}_0'] = self.step_size[current_tran_id][0].detach().cpu()
                    dic[f'ais/step_size_{current_tran_id}_1'] = self.step_size[current_tran_id][1].detach().cpu() 
                    
        return dic

    def sample_ais(self, nb_sample=1, return_log_prob=False, detach_sample = True, return_dic = False):
        """
        Samples from the proposal distribution with different annealed importance sampling steps.
        """
        x_k = self.proposal.sample(nb_sample, return_log_prob=False)
        inv_log_weights = self.proposal.log_prob(x_k).reshape(-1, 1)
        x_k = torch.autograd.Variable(x_k, requires_grad=True)
        dic = {}
        log_acceptance_rate_mean = []
        log_prob_forward_mean = []
        log_prob_backward_mean = []
        log_z = []
        liste_ESS = []
        
        for k in range(1, self.nb_transitions_ais+1):
            for l in range(0, self.nb_step_ais):
                # Compute forward step
                target_density_function = self.get_target(k)
                forward_density = target_density_function(x_k).reshape(-1, 1)
                forward_grad = torch.autograd.grad(forward_density.sum(), x_k, create_graph=True, only_inputs=True)[0]
                eps = torch.randn_like(x_k)
                current_step_size = self.step_size[k-1]
                x_kp1 = x_k.data + current_step_size * forward_grad + math.sqrt(2*current_step_size)*self.sigma_ais * eps

                # Clipping if ever :
                if self.clamp_min_ais is not None or self.clamp_max_ais is not None :
                    x_kp1 = x_kp1.clamp(self.clamp_min_ais, self.clamp_max_ais)
                
                # Compute backward step
                backward_density = target_density_function(x_kp1).reshape(-1, 1)
                backward_grad = torch.autograd.grad(backward_density.sum(), x_kp1, create_graph=True, only_inputs=True)[0]
                eps_reverse = (x_k.data - x_kp1 - current_step_size * backward_grad) / math.sqrt(2*current_step_size)

                # Compute proposal value
                log_prob_backward = torch.distributions.Normal(0, 1).log_prob(eps_reverse.flatten(1)).sum(1, keepdim=True).reshape(-1, 1)
                log_prob_forward = torch.distributions.Normal(0, 1).log_prob(eps.flatten(1)).sum(1, keepdim=True).reshape(-1, 1)

                # Compute acceptance rate
                log_acceptance_rate = log_prob_backward + backward_density - log_prob_forward - forward_density

                
                # Update the log proposal value
                inv_log_weights = inv_log_weights + log_prob_forward - log_prob_backward

                # Update ESS and log_z estimate :
                current_log_z = torch.logsumexp(backward_density - inv_log_weights - math.log(x_k.shape[0]), dim=0) # It's backward density because you want the density at arrival
                current_ESS = 2*torch.logsumexp(backward_density - inv_log_weights, dim=0)- torch.logsumexp(2*backward_density - 2*inv_log_weights, dim=0)
            
                # Update stepsize
                if self.adaptive_step_size:
                    dic_step_size = self.update_stepsize(current_log_z, current_ESS, k-1)
                    dic.update(dic_step_size)

                # Logging intermediate values
                liste_ESS.append(current_ESS.exp().item())
                log_z.append(current_log_z.item())
                log_acceptance_rate_mean.append(log_acceptance_rate.item())
                log_prob_forward_mean.append(log_prob_forward.mean().item())
                log_prob_backward_mean.append(log_prob_backward.mean().item())

                # Update x_k
                x_k.data = x_kp1.data
        
        dic.update({
            "todraw/log_prob_forward": log_prob_forward_mean,
            "todraw/log_prob_backward": log_prob_backward_mean,
            "todraw/log_z": log_z,
            "todraw/ESS": liste_ESS,
            "todraw/log_acceptance_rate": log_acceptance_rate_mean,
            })
    
        return x_k, inv_log_weights, dic


                


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
        if sample_function is None :
            if self.train_ais:
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