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
    def __init__(self, ebm, args_dict, complete_dataset = None, nb_sample_train_estimate= 1024, **kwargs):
        super().__init__(ebm = ebm, args_dict = args_dict, complete_dataset = complete_dataset, nb_sample_train_estimate= nb_sample_train_estimate, **kwargs)
       

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()

        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()
        x = batch['data']
        energy_samples, dic_output = self.ebm.calculate_energy(x)

        
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        dic_output.update(dic)
        if self.ebm.type_z == 'log':
            loss_estimate_z = estimate_log_z
        elif self.ebm.type_z == 'exp':
            loss_estimate_z = estimate_log_z.exp()

        loss_energy = energy_samples.mean()
        loss_total = loss_energy + loss_estimate_z
        self.log('train_loss', loss_total)


        # Update the parameters
        ebm_opt.zero_grad()
        proposal_opt.zero_grad()
        self.manual_backward(loss_total)
        ebm_opt.step()
        proposal_opt.step()
        
        # Add some estimates of the log likelihood with a fixed number of samples
        if self.nb_sample_train_estimate is not None and self.nb_sample_train_estimate > 0 :
            estimate_log_z, _= self.ebm.estimate_log_z(x, nb_sample = self.nb_sample_train_estimate)
            log_likelihood_fix_z = -dic_output['energy'].mean() - estimate_log_z + 1
            self.log('train_log_likelihood_fix_z', log_likelihood_fix_z)
        for key in dic_output:
            self.log(f'train_{key}_mean', dic_output[key].mean().item())
        
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        x = batch['data']
        energy_samples, dic_output = self.ebm.calculate_energy(x)
        return dic_output
    
