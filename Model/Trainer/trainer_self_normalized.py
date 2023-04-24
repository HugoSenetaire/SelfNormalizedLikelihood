import pytorch_lightning as pl
import torch
from ..Utils.optimizer_getter import get_optimizer, get_scheduler
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
        if np.prod(self.args_dict['input_size'])==2:
            nx = 1000
            ny = 1000
            x = np.linspace(self.min_x, self.max_x, nx)
            y = np.linspace(self.min_y, self.max_y, ny)
            xx, yy = np.meshgrid(x, y)
            xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
            xy = torch.from_numpy(xy).float()
            z = (self.ebm.proposal.log_prob(xy)).exp()
            z = z.reshape(nx, ny)

            fig, axs = plt.subplots(1,4,figsize=(20,5,),)
            save_dir = self.args_dict['save_dir']
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if self.example is not None :
                axs[0].contourf(x, y, z, 100)
                axs[0].set_title('Proposal prob')

                axs[1].contourf(x, y, z, 100)
                axs[1].scatter(self.example[:,0], self.example[:,1], s=1, c='r', alpha=0.1)
                axs[1].set_title('Sample dataset')

                axs[2].contourf(x, y, z, 100)
                axs[2].scatter(self.example_proposal[:,0], self.example_proposal[:,1], s=1, c='r', alpha=0.1)
                axs[2].set_title('Sample proposal')

                fig.colorbar(axs[0].contourf(x,y,z,100), cax=axs[3])

                plt.savefig(os.path.join(save_dir, "proposal.png"))
                try :
                    self.logger.log_image(key = "proposal", images = [fig])
                except AttributeError as e :
                    print(e, )
                plt.close()


        

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
        return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(), 'log_prob_samples' : dic_output['log_prob_samples'].flatten()}
    
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
            self.sample()
        
       
    def test_step(self, batch, batch_idx):
        x = batch['data']
        loss, dic_output = self.ebm(x)
        energy = dic_output['energy_batch']
        return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(), 'log_prob_samples' : dic_output['log_prob_samples'].flatten()}
    
    def test_epoch_end(self, outputs):
        sum_energy = torch.cat([outputs[k]['energy']for k in range(len(outputs))]).sum()
        loss_mean = torch.cat([outputs[k]['loss'] for k in range(len(outputs))]).mean()
        aux_samples = self.ebm.proposal.sample(self.args_dict['num_sample_proposal_test']).to(self.device, self.dtype)
        aux_energy = self.ebm.calculate_energy(aux_samples)
        aux_log_prob = self.ebm.proposal.log_prob(aux_samples)
        aux_z = (-aux_energy-aux_log_prob).exp().mean()
        self.log('aux_z', aux_z)
        self.log('test_expectation_energy', sum_energy)
        self.log('test_loss', loss_mean)


    def plot_energy(self, ):
        if self.global_step - self.last_save > self.args_dict['save_energy_every'] :
            save_dir = self.args_dict['save_dir']
            save_dir = os.path.join(save_dir, "contour_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            nx = 1000
            ny = 1000


            x = np.linspace(self.min_x, self.max_x, nx)
            y = np.linspace(self.min_y, self.max_y, ny)
            xx, yy = np.meshgrid(x, y)
            xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
            xy = torch.from_numpy(xy).float()
            z = (-self.ebm.calculate_energy(xy)).exp().detach().cpu().numpy()
            z = z.reshape(nx, ny)


            if self.example is not None :
                fig, axs = plt.subplots(1, 4, figsize=(20,5))

                axs[0].contourf(x,y,z, 100)
                axs[0].set_title('Energy')

                axs[1].contourf(x, y, z, 100)
                axs[1].scatter(self.example[:,0], self.example[:,1], c='r', alpha = 0.1)
                axs[1].set_title('Samples from dataset')

                axs[2].contourf(x, y, z, 100)
                axs[2].scatter(self.example_proposal[:,0], self.example_proposal[:,1], c='r', alpha = 0.1)
                axs[2].set_title('Samples from proposal')
                fig.colorbar(axs[0].contourf(x,y,z, 100), cax=axs[3])

            else :
                fig, axs = plt.subplots(1, 2, figsize = (10,5))

                axs[0].contourf(x,y,z, 100)
                axs[0].set_title('Energy')
                fig.colorbar(axs[0].contourf(x,y,z, 100), cax=axs[1])

            # Add the colorbar to the figure
            plt.savefig(os.path.join(save_dir, "contour_{}.png".format(self.global_step)))
            try :
                self.logger.log_image(key = "contour_{}.png", images = [fig])
            except AttributeError as e :
                print(e, )
            plt.close()
            self.last_save = self.global_step

    def sample(self, num_samples = 1000):
        torch.set_grad_enabled(True)

        if self.global_step - self.last_save_sample > self.args_dict['samples_every'] :
            energy_function = lambda x: self.ebm.calculate_energy(x[0])
            save_dir = self.args_dict['save_dir']
            save_dir = os.path.join(save_dir, "samples_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            samples = nuts_sampler(energy_function, self.ebm.proposal, input_size=self.args_dict['input_size'], num_samples=num_samples, )
            samples = samples[0]
            max_x, min_x = torch.max(samples[:,0],), torch.min(samples[:,0])
            max_y, min_y = torch.max(samples[:,1],), torch.min(samples[:,1])

            x = np.linspace(min_x, max_x, 1000)
            y = np.linspace(min_y, max_y, 1000)
            xx, yy = np.meshgrid(x, y)
            xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
            xy = torch.from_numpy(xy).float()
            z = (-self.ebm.calculate_energy(xy)).exp().detach().cpu().numpy()
            z = z.reshape(1000, 1000)



            fig, axs = plt.subplots(1, 3, figsize=(15,5))
            axs[0].contourf(x, y, z, 100)
            axs[0].set_title('Energy')
            axs[1].contourf(x, y, z, 100)
            axs[1].scatter(samples[:, 0], samples[:, 1], s=1, c='r', alpha = 0.1)
            axs[1].set_title('HMC Samples')
            fig.colorbar(axs[0].contourf(x, y, z, 100), cax=axs[2])

            # Add the colorbar to the figure
            plt.savefig(os.path.join(save_dir, "samples_{}.png".format(self.global_step)))
            try :
                self.logger.log_image(key = "samples_{}.png".format(self.global_step), images = [fig])
            except AttributeError as e :
                print(e, )
            plt.close()
            self.last_save_sample = self.global_step