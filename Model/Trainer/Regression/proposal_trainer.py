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

class ProposalRegressionTrainer(AbstractRegression):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        '''
        Train a only the proposal from the model
        '''
        args_dict['train_proposal'] = True
        super().__init__(ebm, args_dict, complete_dataset = complete_dataset, **kwargs)

    def training_step(self, batch, batch_idx,):
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()

        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()

        x = batch['data']
        y = batch['target']
        log_prob_proposal = self.ebm.log_prob_proposal(x,y)
        proposal_opt.zero_grad()
        proposal_loss = self.proposal_loss(log_prob_proposal=log_prob_proposal, log_estimate_z=None)
        self.manual_backward(proposal_loss, inputs= list(self.ebm.proposal.parameters()))
        proposal_opt.step()

        self.log('train_loss', proposal_loss)
        return proposal_loss


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
        
        
    
    
        

