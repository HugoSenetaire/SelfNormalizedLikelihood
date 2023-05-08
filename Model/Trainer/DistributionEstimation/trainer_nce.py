import pytorch_lightning as pl
import torch
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from ...Sampler import get_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from .abstract_trainer import AbstractDistributionEstimation

class LitSelfNormalized(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal.
    """
    def __init__(self, ebm, args_dict, complete_dataset = None, nb_sample_train_estimate= 1024, **kwargs):
        super().__init__(ebm = ebm, args_dict = args_dict, complete_dataset = complete_dataset, nb_sample_train_estimate= nb_sample_train_estimate, **kwargs)


    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()
        if hasattr(self.ebm.proposal, 'set_x'):
            self.ebm.proposal.set_x(None)
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()
        x = batch['data']

        energy_data, dic_output = self.ebm.calculate_energy(x,)
        energy_data = energy_data.reshape(x.shape[0],)
        
        log_prob_proposal_data = self.ebm.log_prob_proposal(x).reshape(x.shape[0],)
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        estimate_log_z_expanded = estimate_log_z.reshape(1).expand(x.shape[0],)

        nce_numerator = - energy_data - log_prob_proposal_data
        nce_numerator= nce_numerator.reshape(x.shape[0],1)
        nce_denominator = (estimate_log_z_expanded + torch.log(torch.tensor(self.ebm.nb_sample, dtype=x.dtype, device=x.device))).reshape(x.shape[0],1)
        nce_denominator = torch.logsumexp(torch.cat([nce_numerator, nce_denominator], dim=1),dim=1, keepdim=True)

        loss_total = (nce_numerator - nce_denominator).mean()

        # Backward ebm
        ebm_opt.zero_grad()
        proposal_opt.zero_grad()
        self.manual_backward(loss_total, retain_graph=True, )


        # Update the parameters of the proposal
        proposal_opt.zero_grad()
        if self.train_proposal :
            self.log('proposal_log_likelihood', log_prob_proposal_data.mean())
            proposal_loss = self.proposal_loss(log_prob_proposal_data, estimate_log_z,)
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            proposal_opt.step()

        # Update the parameters of the ebm
        ebm_opt.step()
        dic_output.update(dic)

        self.post_train_step_handler(x, dic_output,)

        
        return loss_total
    

