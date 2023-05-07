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

class RegressionTrainerNCE(AbstractRegression):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        super().__init__(ebm, args_dict, complete_dataset = complete_dataset, **kwargs)


    def training_step(self, batch, batch_idx,):
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()

        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()
        dic_output = {}
        x = batch['data']
        y = batch['target']
        nb_sample = self.ebm.nb_sample
        

        energy_data, dic_output = self.ebm.calculate_energy(x, y)
        energy_data = energy_data.reshape(x.shape[0], 1, -1)
        log_prob_proposal_data = self.ebm.log_prob_proposal(x,y).reshape(x.shape[0], 1, -1)
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.ebm.nb_sample)

        nce_numerator = energy_data - log_prob_proposal_data
        nce_numerator= nce_numerator.reshape(x.shape[0], 1, -1)
        nce_denominator = (estimate_log_z + torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))).reshape(x.shape[0], 1, -1)
        nce_denominator = torch.logsumexp(torch.cat([nce_numerator, nce_denominator], dim=1),dim=1)

        loss_total = (nce_numerator - nce_denominator).mean()

        ebm_opt.zero_grad()
        proposal_opt.zero_grad()
        self.manual_backward(loss_total, retain_graph=True, )
        if self.train_proposal :
            proposal_opt.zero_grad()
            proposal_loss = self.proposal_loss(log_prob_proposal=log_prob_proposal_data, log_estimate_z=estimate_log_z)
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            proposal_opt.step()
        ebm_opt.step()
        dic_output.update(dic)

        self.log('train_loss', loss_total)
        for key in dic_output.keys():
            self.log(f'train_{key}', dic_output[key].mean())
        return loss_total

        # if self.ebm.feature_extractor is not None:
            # x_feature = self.ebm.feature_extractor(x)
        # else:
            # x_feature = x
        # # Calculate the energy of the data
        # out_energy = self.energy(x_feature, y)
        # dic_output['f_theta'] = out_energy
        # if self.ebm.explicit_bias is not None :
        #     b_samples = self.ebm.explicit_bias(x_feature).reshape(out_energy.shape)
        #     out_energy = out_energy + b_samples
        #     dic_output['b'] = b_samples

        # if self.ebm.base_dist is not None :
        #     if len(x.shape) == 1:
        #         x = x.unsqueeze(0)
        #     base_dist_log_prob = self.base_dist.log_prob(x, y).view(x.size(0), -1).sum(1).unsqueeze(1)
        #     dic_output['base_dist_log_prob'] = base_dist_log_prob
        # else :
        #     base_dist_log_prob = torch.zeros_like(out_energy)
        # current_energy_data = out_energy - base_dist_log_prob
        # dic_output['energy'] = current_energy_data

        # log_prob_proposal_data = self.ebm.proposal.log_prob(x_feature,y)

        # # Get samples from proposal
        # samples_y_proposal = self.ebm.proposal.sample(x_feature, nb_sample)
        # x_feature_expanded = x.unsqueeze(1).expand(-1, nb_sample, -1).reshape(-1, *x_feature.shape[1:])

        # # Calculate the energy of the samples
        # out_energy_samples = self.energy(x_feature_expanded, samples_y_proposal)
        # dic_output['f_theta_samples'] = out_energy_samples
        # if self.ebm.explicit_bias is not None :
        #     b = self.ebm.explicit_bias(x_feature_expanded).reshape(out_energy_samples.shape)
        #     out_energy_samples = out_energy_samples + b
        #     dic_output['b_samples'] = b

        # if self.ebm.base_dist is not None :
        #     if len(x.shape) == 1:
        #         x = x.unsqueeze(0)
        #     base_dist_log_prob_samples = self.base_dist.log_prob(x, y).view(x.size(0), -1).sum(1).unsqueeze(1)
        #     dic_output['base_dist_log_prob_samples'] = base_dist_log_prob_samples
        # else :
        #     base_dist_log_prob_samples = torch.zeros_like(out_energy_samples)
        # current_energy_samples = out_energy_samples - base_dist_log_prob_samples
        # dic_output['energy_samples'] = current_energy_samples

        # log_prob_proposal_sample = self.ebm.proposal.log_prob(x_feature_expanded, samples_y_proposal).reshape(x.shape[0], nb_sample, -1).sum(2)
        # current_energy_samples = current_energy_samples.reshape(x.shape[0], nb_sample, -1).sum(2)

        # nce_loss = current_energy_data - log_prob_proposal_data - torch.logsumexp(current_energy_samples - log_prob_proposal_sample, dim=1) + torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))
        # loss_total = nce_loss.mean()

        # estimate_log_z = current_energy_samples - log_prob_proposal_sample


        # Update the parameters
      


        
    
    
        

