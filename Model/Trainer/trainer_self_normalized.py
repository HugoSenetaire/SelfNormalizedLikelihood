import pytorch_lightning as pl
import torch
from ..Utils.optimizer_getter import get_optimizer, get_scheduler
import yaml


class LitSelfNormalized(pl.LightningModule):
    def __init__(self, ebm, args_dict, **kwargs):
        super().__init__()
        self.ebm = ebm
        self.args_dict = args_dict
        self.hparams.update(args_dict)
        # Write args dict to save dir
        # with open(self. + '/args_dict.yaml', 'w') as f:
            # yaml.dump(args_dict, f)
        # self.hparams = args_dict
        # self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
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
        sum_energy = torch.cat([outputs[k]['energy'] for k in range(len(outputs))]).sum()
        loss_mean = torch.cat([outputs[k]['loss'] for k in range(len(outputs))]).mean()
        aux_samples = self.ebm.proposal.sample(self.args_dict['num_sample_proposal_test']).to(self.device, self.dtype)
        aux_energy = self.ebm.energy(aux_samples)
        aux_log_prob = self.ebm.proposal.log_prob(aux_samples)
        aux_z = (-aux_energy-aux_log_prob).exp().mean()
        self.log('aux_z', aux_z)
        self.log('val_sum_energy_input', sum_energy)
        self.log('val_loss', loss_mean)

    def test_step(self, batch, batch_idx):
        x = batch['data']
        loss, dic_output = self.ebm(x)
        energy = dic_output['energy_batch']
        return {'energy' : energy.flatten(), 'loss' : loss.flatten(), 'energy_samples' : dic_output['energy_samples'].flatten(), 'log_prob_samples' : dic_output['log_prob_samples'].flatten()}
    
    def test_epoch_end(self, outputs):
        sum_energy = torch.cat([outputs[k]['energy']for k in range(len(outputs))]).sum()
        loss_mean = torch.cat([outputs[k]['loss'] for k in range(len(outputs))]).mean()
        aux_samples = self.ebm.proposal.sample(self.args_dict['num_sample_proposal_test']).to(self.device, self.dtype)
        aux_energy = self.ebm.energy(aux_samples)
        aux_log_prob = self.ebm.proposal.log_prob(aux_samples)
        aux_z = (-aux_energy-aux_log_prob).exp().mean()
        self.log('aux_z', aux_z)
        self.log('test_expectation_energy', sum_energy)
        self.log('test_loss', loss_mean)

    