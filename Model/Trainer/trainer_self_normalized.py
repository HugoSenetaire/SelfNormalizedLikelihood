import pytorch_lightning as pl
import torch
from ..Utils.optimizer_getter import get_optimizer, get_scheduler
from ..Utils.plot_utils import plot_energy_2d, plot_images
from ..Sampler import nuts_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml


class LitSelfNormalized(pl.LightningModule):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        super().__init__()
        self.ebm = ebm
        self.args_dict = args_dict
        self.hparams.update(args_dict)
        self.last_save = -float('inf')
        self.last_save_sample = -float('inf')
        self.complete_dataset = complete_dataset
        self.num_samples = 0
        self.save_hyperparameters()
        self.initialize_examples()
        self.proposal_visualization()



    def initialize_examples(self,):
        if self.complete_dataset is not None :
            indexes_to_print = np.random.choice(len(self.complete_dataset.dataset_train), 10000)
            self.example = torch.cat([self.complete_dataset.dataset_train.__getitem__(i)[0] for i in indexes_to_print], dim=0)
            self.min_x, self.max_x = min(torch.min(self.example[:,0],), -3), max(torch.max(self.example[:,0]), 3)
            self.min_y, self.max_y = min(torch.min(self.example[:,1],), -3), max(torch.max(self.example[:,1]), 3)
        else :
            self.example = None
            self.min_x, self.max_x, self.min_y, self.max_y = -3, 3, -3, 3
        if self.ebm.proposal is not None :
            self.example_proposal = self.ebm.proposal.sample(1000)
            self.min_x, self.max_x = min(torch.min(self.example_proposal[:,0],), self.min_x), max(torch.max(self.example_proposal[:,0]), self.max_x)
            self.min_y, self.max_y = min(torch.min(self.example_proposal[:,1],), self.min_y), max(torch.max(self.example_proposal[:,1]), self.max_y)
        else :
            self.example_proposal = None

    def proposal_visualization(self):
        if np.prod(self.args_dict['input_size']) == 2:
            energy_function = lambda x: -self.ebm.proposal.log_prob(x)
            save_dir = self.args_dict['save_dir']
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_energy_2d(self, energy_function=energy_function, save_dir=save_dir, samples = [self.example, self.example_proposal], samples_title=['Samples from dataset','Samples from proposal'], name='proposal',)

    def training_step(self, batch, batch_idx):
        if self.args_dict["switch_mode"] is not None and self.global_step == self.args_dict["switch_mode"]:
            self.ebm.switch_mode()
        x = batch['data']
        loss, dic_output = self.ebm(x)
        loss = loss.mean()
        for key in dic_output:
            self.log(f'train_{key}', dic_output[key].mean().item())
        return loss
        
    def configure_optimizers(self):
        res = {}
        res['optimizer'] = get_optimizer(model = self.ebm, args_dict = self.args_dict)
        scheduler = get_scheduler(args_dict = self.args_dict, model = self.ebm, optim = res['optimizer'])
        if scheduler is not None:
            res['lr_scheduler'] = scheduler
        return res        

    def validation_step(self, batch, batch_idx,):
        x = batch['data']
        loss, dic_output = self.ebm(x)
        energy = dic_output['energy_batch']
        try :
            return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(), 'log_prob_samples' : dic_output['log_prob_samples'].flatten()}
        except KeyError as e :
            return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(),}
        
    def validation_epoch_end(self, outputs):
        mean_energy = torch.cat([outputs[k]['energy'] for k in range(len(outputs))]).mean()
        loss_mean = torch.cat([outputs[k]['loss'] for k in range(len(outputs))]).mean()


        aux_samples = self.ebm.proposal.sample(self.args_dict['num_sample_proposal_test']).to(self.device, self.dtype)
        aux_energy = self.ebm.calculate_energy(aux_samples).flatten()
        aux_log_prob = self.ebm.proposal.log_prob(aux_samples)

        estimated_z_SN = (-aux_energy-aux_log_prob).exp().mean()
        self.log('estimated_z_SN', estimated_z_SN)
        estimated_z_IW = torch.logsumexp(-aux_energy-aux_log_prob, dim = 0) - torch.log(torch.tensor(self.args_dict['num_sample_proposal_test'], dtype = torch.float32))
        self.log('estimated_z_IW', estimated_z_IW)
        
        self.log('val_mean_energy_input', mean_energy)

        self.log('val_loss', loss_mean)
        if np.prod(self.args_dict['input_size']) == 2:
            self.plot_energy()
        self.plot_samples()
        
       
    def test_step(self, batch, batch_idx):
        x = batch['data']
        loss, dic_output = self.ebm(x)
        energy = dic_output['energy_batch']
        try :
            return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(), 'log_prob_samples' : dic_output['log_prob_samples'].flatten()}
        except KeyError as e:
            return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(),}
        
        
    def test_epoch_end(self, outputs):
        sum_energy = torch.cat([outputs[k]['energy']for k in range(len(outputs))]).mean()
        loss_mean = torch.cat([outputs[k]['loss'] for k in range(len(outputs))]).mean()
        aux_samples = self.ebm.proposal.sample(self.args_dict['num_sample_proposal_test']).to(self.device, self.dtype)
        aux_energy = self.ebm.calculate_energy(aux_samples).flatten()
        aux_log_prob = self.ebm.proposal.log_prob(aux_samples)

        estimated_z_SN = (-aux_energy-aux_log_prob).exp().mean()
        self.log('estimated_z_SN', estimated_z_SN)
        estimated_z_IW = torch.logsumexp(-aux_energy-aux_log_prob, dim = 0) - torch.log(torch.tensor(self.args_dict['num_sample_proposal_test'], dtype = torch.float32))
        self.log('estimated_z_IW', estimated_z_IW)
        
        self.log('test_mean_energy_input', sum_energy)
        self.log('test_loss', loss_mean)


    def plot_energy(self, ):
        if np.prod(self.args_dict['input_size']) == 2:
            if self.global_step - self.last_save > self.args_dict['save_energy_every'] :
                save_dir = self.args_dict['save_dir']
                save_dir = os.path.join(save_dir, "contour_energy")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_energy_2d(self, save_dir=save_dir, samples = [self.example, self.example_proposal], samples_title=['Samples from dataset','Samples from proposal'], name='contour', step=self.global_step)
                self.last_save = self.global_step

    def samples(self, num_samples = 1000):
        energy_function = lambda x: self.ebm.calculate_energy(x[0])
        samples = nuts_sampler(energy_function, self.ebm.proposal, input_size=self.args_dict['input_size'], num_samples=num_samples, )
        return samples[0]


    def plot_samples(self, num_samples = 1000):
        torch.set_grad_enabled(True)
        if self.global_step - self.last_save_sample > self.args_dict['samples_every'] :
            save_dir = self.args_dict['save_dir']
            save_dir = os.path.join(save_dir, "samples_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            samples = self.samples(num_samples = num_samples)

            if np.prod(self.args_dict['input_size']) == 2:
                plot_energy_2d(self, save_dir=save_dir, samples = [samples], samples_title=['HMC samples'], name='samples', step=self.global_step)
            elif len(self.args_dict['input_size']) == 3 :
                plot_images(self, save_dir=save_dir, samples = [samples], samples_title=['HMC samples'], name='samples', step=self.global_step)
            else :
                raise NotImplementedError
            self.last_save_sample = self.global_step


