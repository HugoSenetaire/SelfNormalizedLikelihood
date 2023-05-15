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
import torch.autograd

class ScoreMatchingTrainer(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal.
    """
    def __init__(self, ebm, args_dict, complete_dataset = None, nb_sample_train_estimate= 1024, **kwargs):
        super().__init__(ebm = ebm, args_dict = args_dict, complete_dataset = complete_dataset, nb_sample_train_estimate= nb_sample_train_estimate, **kwargs)


    def exact_score_matching(self, data,):
        data = data.flatten(1)
        data.requires_grad_(True)
        energy_data, dic = self.ebm.calculate_energy(data)
        logp = -energy_data.reshape(data.shape[0]).sum(-1)  # logp(x)
        grad1 = torch.autograd.grad(logp, data, create_graph=True)[0]
        loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
        loss2 = torch.zeros(data.shape[0], device=data.device)

        iterator = range(data.shape[1])

        for i in iterator:
            if self.training:
                grad = torch.autograd.grad(grad1[:, i].sum(), data, create_graph=True, retain_graph=True)[0][:, i]
            if not self.training:
                grad = torch.autograd.grad(grad1[:, i].sum(), data, create_graph=False, retain_graph=True)[0][:, i]
                grad = grad.detach()
            loss2 += grad

        loss = loss1 + loss2

        if not self.training:
            loss = loss.detach()

        return loss, dic


    def training_step(self, batch, batch_idx):
        ebm_opt, proposal_opt = self.optimizers_perso()

        if (
            self.args_dict["switch_mode"] is not None
            and self.global_step == self.args_dict["switch_mode"]
        ):
            self.ebm.switch_mode()
        x = batch['data']
        if hasattr(self.ebm.proposal, 'set_x'):
            self.ebm.proposal.set_x(x)


        loss_total, dic_output = self.exact_score_matching(x)
        self.log("train_loss", loss_total.mean())
        ebm_opt.zero_grad()
        self.manual_backward(loss_total.mean(), retain_graph=False, )
        ebm_opt.step()

        ebm_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        estimate_log_z = estimate_log_z.mean()
        dic_output.update(dic)


        # Update the parameters of the proposal
        if self.train_proposal :
            proposal_opt.zero_grad()
            log_prob_proposal_data = self.ebm.proposal.log_prob(
                x,
            )
            self.log('proposal_log_likelihood', log_prob_proposal_data.mean())
            proposal_loss = self.proposal_loss(log_prob_proposal_data, estimate_log_z,)
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            proposal_opt.step()

        self.post_train_step_handler(x, dic_output,)

        
        return loss_total
    

