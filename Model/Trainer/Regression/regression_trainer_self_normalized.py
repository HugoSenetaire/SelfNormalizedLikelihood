import pytorch_lightning as pl
import torch
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from ...Sampler import get_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from .abstract_regression_trainer import AbstractRegression

class RegressionTrainerSelfNormalized(AbstractRegression):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        super().__init__(ebm, args_dict, complete_dataset = complete_dataset, **kwargs)


    def training_step(self, batch, batch_idx,):
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()

        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()

        x = batch['data']
        y = batch['target']
        energy_data, dic_output = self.ebm.calculate_energy(x, y)
        energy_data.reshape(x.shape[0], 1, -1)
        log_prob_proposal_data = self.ebm.log_prob_proposal(x,y).reshape(x.shape[0], 1, -1)
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        dic_output.update(dic)
        dic_output.update({'log_prob_proposal_data': log_prob_proposal_data.mean()})
        estimate_log_z = estimate_log_z.reshape(x.shape[0], 1, -1).exp()
        loss_total = (energy_data + estimate_log_z).mean()

        # Update the parameters
        ebm_opt.zero_grad()
        proposal_opt.zero_grad()
        self.manual_backward(loss_total, retain_graph=True, )
        if self.train_proposal :
            proposal_opt.zero_grad()
            proposal_loss = self.proposal_loss(log_prob_proposal=log_prob_proposal_data, log_estimate_z=estimate_log_z)
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            self.log('train_proposal_loss', proposal_loss.mean())
            dic_output.update({"proposal_loss" : proposal_loss.mean()})
            proposal_opt.step()
        ebm_opt.step()
        dic_output.update(dic)

        self.log('train_loss', loss_total)
        for key in dic_output.keys():
            self.log(f'train_{key}', dic_output[key].mean())
        return loss_total



    
    
        

