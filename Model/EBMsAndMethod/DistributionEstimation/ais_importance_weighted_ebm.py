import itertools
import math

import torch
import torch.distributions as distributions
import torch.nn as nn
import numpy as np

from ...Sampler.utils_sampler.clip_sampler import clip_data, clip_grad
from .importance_weighted_ebm import ImportanceWeightedEBM


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

        self.epsilons = [torch.tensor(self.step_size_ais) for _ in range(self.nb_transitions_ais)]
        if self.variance_sensitive_step:
            self.gamma_0 = [torch.tensor(0.1) for _ in range(self.nb_transitions_ais)] 
        

        super().__init__(
            f_theta=f_theta,
            proposal=proposal,
            base_dist=base_dist,
            explicit_bias=explicit_bias,
            cfg_ebm=cfg_ebm,
            nb_sample_init_bias=nb_sample_init_bias,
            )


        


    def get_target(self, k,):
        """
        Get the annealed function to target at iteration k of the AIS algorithm.
        """
 
        value = min(1.0, k / (self.nb_transitions_ais))
        return lambda x: -self.calculate_energy(x, use_base_dist=True)[0].flatten() * value + self.proposal.log_prob(x).flatten() * (1-value)

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
                    
                    self.epsilons[current_tran_id] = self.alpha_stepsize_decrease * self.epsilons[current_tran_id]
                else:
                    self.epsilons[current_tran_id] = self.alpha_stepsize_increase * self.epsilons[current_tran_id]

                self.epsilons[current_tran_id] = self.epsilons[current_tran_id].clamp(self.step_size_min, self.step_size_max)
                dic[f'ais/epsilons_{current_tran_id}'] = self.epsilons[current_tran_id].detach().cpu()

                    # self.transitions[l].log_stepsize.data = torch.tensor(np.log(self.epsilons[l]), dtype=torch.float32,
                                                                        #  device=self.transitions[l].log_stepsize.device)
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
                    self.epsilons[current_tran_id] = 0.9 * self.epsilons[current_tran_id] + 0.1 * self.gamma_0[
                        current_tran_id] / (gradient_std + 1.)

                    if accept_rate[current_tran_id].mean() < self.acceptance_rate_target:
                        self.gamma_0[current_tran_id] *= 0.99
                    else:
                        self.gamma_0[current_tran_id] *= 1.02
                    
                    dic[f'ais/gamma_0_{current_tran_id}'] = self.gamma_0[current_tran_id].detach().cpu()
                    dic[f'ais/acceptrate_{current_tran_id}'] = accept_rate[current_tran_id].mean().detach().cpu()

                    self.epsilons[current_tran_id] = self.epsilons[current_tran_id].clamp(self.step_size_min, self.step_size_max)
                    # if self.epsilons
                    dic[f'ais/epsilons_{current_tran_id}_0'] = self.epsilons[current_tran_id][0].detach().cpu()
                    dic[f'ais/epsilons_{current_tran_id}_1'] = self.epsilons[current_tran_id][1].detach().cpu() 
                    
       
                    
        # else:
            # pass

       
        

        return dic


    def sample_ais(self, nb_sample=1, return_log_prob=False, detach_sample = True, return_full_list=False, return_dic = False):
        """
        Samples from the proposal distribution with different annealed importance sampling steps.
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
        listes_samples_proposal = [samples_proposal]
        if return_log_prob :
            liste_samples_proposal_log_prob = [samples_proposal_log_prob.reshape(-1, 1)]
        liste_acceptance_rate = []
        dic = {}

        for k in range(1,self.nb_transitions_ais+1):
            eps = torch.randn_like(samples_proposal)
            target_density = self.get_target(k)
            density_init = target_density(listes_samples_proposal[-1]).reshape(-1, 1)
            forward_grad = torch.autograd.grad(density_init.sum(), listes_samples_proposal[-1], create_graph=True, only_inputs=True)[0]
            

            update = torch.sqrt(2 * self.epsilons[k-1]) * eps + self.epsilons[k-1] * forward_grad
            listes_samples_proposal.append(listes_samples_proposal[-1] + update)
            if return_log_prob :
                density_step = target_density(listes_samples_proposal[-1]).reshape(-1, 1)
                backward_grad = torch.autograd.grad(density_step.sum(), listes_samples_proposal[-1], create_graph=True, only_inputs=True)[0]
                eps_reverse = (listes_samples_proposal[-2] - listes_samples_proposal[-1] - self.epsilons[k-1] * backward_grad) / torch.sqrt(2 * self.epsilons[k-1])
                backward_prob = dist.log_prob(eps_reverse.flatten(1)).sum(1,keepdim=True).reshape(-1, 1)
                forward_prob = dist.log_prob(eps.flatten(1)).sum(1, keepdim=True).reshape(-1, 1)
                liste_samples_proposal_log_prob.append(liste_samples_proposal_log_prob[-1] - backward_prob + forward_prob)
                liste_acceptance_rate.append((backward_prob + density_step - forward_prob - density_init).exp().mean(0))
                
                dic_step_size = self.update_stepsize(accept_rate=liste_acceptance_rate, current_tran_id=k-1, current_gradient_batch=forward_grad)
                dic.update(dic_step_size)
                if self.nb_transitions_ais <11 :
                    dic[f'ais/diff_step_{k-1}'] = (backward_prob - forward_prob).mean().detach().cpu()
                    dic[f'ais/diff_density_{k-1}'] = (density_step - density_init).mean().detach().cpu()
                    dic[f'ais/forward_prob_{k-1}'] = forward_prob.mean().detach().cpu()
                    dic[f'ais/backward_prob_{k-1}'] = backward_prob.mean().detach().cpu()
                    dic[f'ais/samples_proposal_log_prob_{k-1}'] = liste_samples_proposal_log_prob[-1].mean().detach().cpu()
                    dic[f'ais/density_step_{k-1}'] = density_step.mean().detach().cpu()
                    dic[f'ais/density_init_{k-1}'] = density_init.mean().detach().cpu()
                    dic[f'ais/log_accept_rate_{k-1}'] = (backward_prob + density_step - forward_prob - density_init).mean()
                listes_samples_proposal[-1] = listes_samples_proposal[-1].detach()
                liste_samples_proposal_log_prob[-1] = liste_samples_proposal_log_prob[-1].detach()
                listes_samples_proposal[-1].requires_grad_(True)
                    
            else :
                listes_samples_proposal[-1] = listes_samples_proposal[-1].detach()
                listes_samples_proposal[-1].requires_grad_(True)

        if return_log_prob :               
            if self.nb_transitions_ais <10 :
                for k in range(self.nb_transitions_ais):
                    dic['ais/accept_rate'+str(k)] = liste_acceptance_rate[k].detach().cpu()
            else :
                acceptance_rate_stacked = torch.stack(liste_acceptance_rate)
                dic['ais/accept_rate_mean'] = acceptance_rate_stacked.mean(0).detach().cpu()
                dic['ais/accept_rate_max'] = acceptance_rate_stacked.max(0)[0].detach().cpu()
                dic['ais/accept_rate_min'] = acceptance_rate_stacked.min(0)[0].detach().cpu()
                    

        
        to_return = []
        if return_full_list:
            to_return.append(listes_samples_proposal)
            if return_log_prob :
                to_return.append(liste_samples_proposal_log_prob)
        else :
            to_return.append(listes_samples_proposal[-1])
            if return_log_prob :
                to_return.append(liste_samples_proposal_log_prob[-1])
        if return_dic :
            to_return.append(dic)
        if len(to_return) == 1:
            to_return = to_return[0]
        return to_return

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