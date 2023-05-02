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


    def training_step(self, batch, batch_idx):
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()

        
        x = batch['data']
        y = batch['target']
        energy_data, dic_output = self.ebm.calculate_energy(x, y)
        
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        dic_output.update(dic)
        if self.ebm.type_z == 'exp':
            estimate_log_z = estimate_log_z.exp() - 1

        loss_total = (energy_data + estimate_log_z).mean()
        self.log('train_loss', loss_total)
        for key in dic_output.keys():
            self.log(f'train_{key}', dic_output[key].mean())
            # print(f'train_{key}', dic_output[key].mean().item())
        return loss_total


    def validation_step(self, batch, batch_idx,):
        x = batch['data']
        y = batch['target']
        energy_data, dic_output = self.ebm.calculate_energy(x,y)

        energy_function = lambda x: -self.ebm.proposal.log_prob(x)
        save_dir = self.args_dict['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        dic_output.update(dic)
        if self.ebm.type_z == 'exp':
            estimate_log_z = estimate_log_z.exp() - 1


        dic_output['loss'] = (energy_data + estimate_log_z)
        dic_output['likelihood'] = - dic_output['loss']
        loss_total = (energy_data + estimate_log_z).mean()
        self.log('val_loss', loss_total)

        return dic_output
        
        
    
    
        

