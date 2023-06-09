import pytorch_lightning as pl
import torch
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_1d_1d_regression, plot_energy_image_1d_regression
from ...Utils.proposal_loss import log_prob_kl_loss_regression, kl_loss_regression, log_prob_loss_regression
from ...Sampler import get_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import itertools


class AbstractRegression(pl.LightningModule):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        super().__init__()
        self.ebm = ebm
        self.args_dict = args_dict
        self.hparams.update(args_dict)
        self.last_save = -float('inf')
        self.num_samples_val = self.args_dict['num_sample_proposal_val']

        if np.prod(self.args_dict['input_size_x']) == 1 :
            self.input_type_x = '1d'
        elif len(self.args_dict['input_size_x']) == 3 :
            self.input_type_x = 'images'
        else :
            self.input_type_x = 'other'

        if np.prod(self.args_dict['input_size_y']) == 1 :
            self.input_type_y = '1d'
        elif np.prod(self.args_dict['input_size_y']) == 2 :
            self.input_type_y = '2d'
        else :
            self.input_type_y = 'other'

        if hasattr( complete_dataset, 'transform_back',):
            self.transform_back = complete_dataset.transform_back
        else :
            self.transform_back = None

        self.initialize_examples(complete_dataset)
        self.proposal_visualization()
        self.train_proposal = self.args_dict['train_proposal']

        if not self.train_proposal :
            for param in self.ebm.proposal.parameters():
                param.requires_grad = False
        else :
            for param in self.ebm.proposal.parameters():
                param.requires_grad = True

        self.proposal_loss_name = args_dict['proposal_loss_name']
        if self.proposal_loss_name == 'log_prob' :
            self.proposal_loss = log_prob_loss_regression
        elif self.proposal_loss_name == 'kl' :
            self.proposal_loss = kl_loss_regression
        elif self.proposal_loss_name == 'log_prob_kl' :
            self.proposal_loss = log_prob_kl_loss_regression
        else :
            raise ValueError('Proposal loss name not recognized')

        self.test_name = None
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx): 
        raise NotImplementedError
    
    
    def validation_step(self, batch, batch_idx, name = 'val'):
        x = batch['data']
        y = batch['target']
        energy_data, dic_output = self.ebm.calculate_energy(x,y)
        energy_data = energy_data.reshape(x.shape[0],)
        if self.ebm.feature_extractor is not None:
            x_feature = self.ebm.feature_extractor(x)
        else :
            x_feature = x
        log_prob_proposal_data = self.ebm.proposal.log_prob(x_feature,y).reshape(x.shape[0],)
        save_dir = self.args_dict['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        estimate_log_z, dic=self.ebm.estimate_log_z(x, nb_sample = self.num_samples_val,)
        estimate_log_z = estimate_log_z.reshape(x.shape[0],)
        dic_output.update(dic)
        dic_output['loss_self_normalized'] = (energy_data + estimate_log_z.exp() - 1).reshape(x.shape[0])
        dic_output['log_likelihood_self_normalized'] = - dic_output['loss_self_normalized'].reshape(x.shape[0])
        dic_output['loss_importance'] = (energy_data + estimate_log_z).reshape(x.shape[0])
        dic_output['log_likelihood_importance'] = - dic_output['loss_importance'].reshape(x.shape[0])
        dic_output['log_likelihood_proposal'] = log_prob_proposal_data.reshape(x.shape[0])
        loss_total = (energy_data + estimate_log_z).mean()
        self.log(f'{name}_loss', loss_total)

        return dic_output
    
    def resample_proposal(self):
        if self.ebm.proposal is not None :
            if self.input_type_y != 'other':
                if self.input_type_x == 'images':
                    self.example_proposal_x = self.example_x_default.to(dtype=self.dtype, device=self.device)
                    self.example_proposal_y = self.ebm.sample_proposal(self.example_proposal_x, 100).reshape(-1, 100, np.prod(self.args_dict['input_size_y'])).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_x = self.example_proposal_x.unsqueeze(1).expand(-1, 100, *self.args_dict['input_size_x']).reshape(-1, 100, *self.args_dict['input_size_x']).to(dtype=self.dtype)
                    self.min_y, self.max_y = min(torch.min(self.example_proposal_y), self.min_y_original), max(torch.max(self.example_proposal_y), self.max_y_original)
                elif self.input_type_x == '1d' :
                    self.example_proposal_x = torch.arange(self.min_x_original, self.max_x_original, (self.max_x_original-self.min_x_original)/100).reshape(-1, 1).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_y = self.ebm.sample_proposal(self.example_proposal_x, 100).reshape(-1, 100, np.prod(self.args_dict['input_size_y'])).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_x = self.example_proposal_x.unsqueeze(1).expand(-1, 100, -1).reshape(-1, 100, 1)
                    self.min_y, self.max_y = min(torch.min(self.example_proposal_y), self.min_y_original), max(torch.max(self.example_proposal_y), self.max_y_original)
                else :
                    return None
    
    def initialize_examples(self, complete_dataset):
        if any([self.input_type_x=='other', self.input_type_y == 'other']):
            return None
        self.min_x_original, self.max_x_original, self.min_y_original, self.max_y_original = 0, 1, 0, 1
        self.min_x, self.max_x, self.min_y, self.max_y = self.min_x_original, self.max_x_original, self.min_y_original, self.max_y_original

        if complete_dataset is not None :
            if self.input_type_x == '1d' :
                nb_samples_max = 10000
            elif self.input_type_x == 'images' :
                nb_samples_max = 10
            indexes_to_print = np.random.choice(len(complete_dataset.dataset_train), min(nb_samples_max, len(complete_dataset.dataset_train)), replace=False)
            
            self.example_x = torch.cat([complete_dataset.dataset_train.__getitem__(i)[0] for i in indexes_to_print], dim=0).reshape(-1, *self.args_dict['input_size_x']).to(dtype=self.dtype)
            self.example_x_default = self.example_x.clone()
            self.example_x = self.example_x.unsqueeze(1).expand(-1, 100, *self.args_dict['input_size_x'])
            self.example_y = torch.cat([complete_dataset.dataset_train.__getitem__(i)[1].unsqueeze(0) for i in indexes_to_print], dim=0).reshape(-1, np.prod(self.args_dict['input_size_y'])).unsqueeze(1).expand(-1, 100, *self.args_dict['input_size_y'])
            self.example_y = self.example_y.to(dtype=self.dtype)
            if self.input_type_x == '1d' :
                self.min_x_original, self.max_x_original = min(torch.min(self.example_x), self.min_x_original), max(torch.max(self.example_x), self.max_x_original)
            if self.input_type_y == '1d' or self.input_type_y == '2d':
                self.min_y_original, self.max_y_original = min(torch.min(self.example_y), self.min_y_original), max(torch.max(self.example_y), self.max_y_original)
        else :
            self.example_x, self.example_y = None, None




    def proposal_visualization(self, step = ''):

        if self.ebm.proposal is not None :
            self.resample_proposal()
            energy_function = lambda x,y: -self.ebm.log_prob_proposal(x,y)
            save_dir = self.args_dict['save_dir']
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if np.prod(self.args_dict['input_size_x']) == 1 and np.prod(self.args_dict['input_size_y']) == 1:
                plot_energy_1d_1d_regression(self,
                                            save_dir=save_dir,
                                            name='proposal',
                                            energy_function=energy_function,
                                            samples_x=[self.example_x, self.example_proposal_x],
                                            samples_y=[self.example_y, self.example_proposal_y],
                                            samples_title=['data', 'proposal'],
                                            step=step,

                                            )
            elif self.input_type_x == 'images' and self.input_type_y == '1d':
                plot_energy_image_1d_regression(self,
                                                save_dir=save_dir,
                                                name='proposal',
                                                energy_function=energy_function,
                                                samples_x=self.example_x,
                                                samples_y=[self.example_y, self.example_proposal_y],
                                                samples_title=['data', 'proposal'],
                                                step=step,
                                                transform_back=self.transform_back,
                                                )
        
    def configure_optimizers(self):
        parameters_ebm = [child.parameters() for name,child in self.ebm.named_children() if name != 'proposal']
        parameters_proposal = [self.ebm.proposal.parameters()] if self.ebm.proposal is not None else []
        
        ebm_opt = get_optimizer( args_dict = self.args_dict, list_parameters_gen = parameters_ebm)
        proposal_opt = get_optimizer( args_dict = self.args_dict, list_parameters_gen = parameters_proposal)

        ebm_sch = get_scheduler(args_dict = self.args_dict, optim = ebm_opt)
        proposal_sch = get_scheduler(args_dict = self.args_dict, optim = proposal_opt)
        if ebm_sch is not None and proposal_sch is not None :
            return [ebm_opt, proposal_opt], [ebm_sch, proposal_sch]      
        elif ebm_sch is not None :
            return [ebm_opt, proposal_opt], ebm_sch
        elif proposal_sch is not None :
            return [ebm_opt, proposal_opt], proposal_sch
        else :
            return [ebm_opt, proposal_opt]

        
    def update_dic_logger(self, outputs, name = 'val_'):
        list_keys = list(outputs[0].keys())
        dic_output = {}
        for key in list_keys:
            try :
                dic_output[name+key+'_mean'] = torch.cat([outputs[k][key] for k in range(len(outputs))]).mean()
            except RuntimeError:
                try :
                    dic_output[name+key+'_mean'] =  torch.cat([outputs[k][key].unsqueeze(0) for k in range(len(outputs))]).mean()
                except RuntimeError:
                    print(key)
        for key in dic_output:
            self.log(key, dic_output[key])

    def validation_epoch_end(self, outputs):
        self.update_dic_logger(outputs, name = 'val_')
        self.plot_energy()
        if self.train_proposal :
            self.proposal_visualization(step = self.global_step)


        
       
    def test_step(self, batch, batch_idx):
        self.num_samples_val = self.args_dict['num_sample_proposal_test']
        if self.test_name is None :
            return self.validation_step(batch, batch_idx, name = 'test') # Regular testing
        else :
            return self.validation_step(batch, batch_idx, name = self.test_name) # Testing with a specific name for specific proposal evaluation
        
        
    def test_epoch_end(self, outputs):
        if self.test_name is None :
            self.update_dic_logger(outputs, name = 'test_') # Regular testing
        else :
            self.update_dic_logger(outputs, name = self.test_name+'_') # Testing with a specific name for specific proposal evaluation
        
    
    def plot_energy(self, ):
        if self.global_step - self.last_save > self.args_dict['save_energy_every'] :
            self.last_save = self.global_step
            self.resample_proposal()
            save_dir = self.args_dict['save_dir']
            save_dir = os.path.join(save_dir, "contour_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ebm_function_list = [lambda x,y: self.ebm.calculate_energy(x,y)[1]['f_theta'],]
            ebm_function_name = ['f_theta',]
            if hasattr(self.ebm, 'explicit_bias') and self.ebm.explicit_bias is not None :
                ebm_function_list.append(lambda x,y: self.ebm.calculate_energy(x,y)[1]['b'],)
                ebm_function_name.append('b')
            if self.input_type_x == '1d' and self.input_type_y == '1d' :
                plot_energy_1d_1d_regression(self,
                                        save_dir=save_dir,
                                        samples_x = [self.example_x,self.example_proposal_x],
                                        samples_y = [self.example_y,self.example_proposal_y],
                                        samples_title=['Samples from dataset','Samples from proposal'],
                                        name='contour',
                                        step=self.global_step)
                for ebm_function, ebm_name in zip(ebm_function_list, ebm_function_name):
                    plot_energy_1d_1d_regression(self,
                                            save_dir=save_dir,
                                            energy_function=ebm_function,
                                            samples_x = [self.example_x,self.example_proposal_x],
                                            samples_y = [self.example_y,self.example_proposal_y],
                                            samples_title=['Samples from dataset','Samples from proposal'],
                                            name=ebm_name,
                                            step=self.global_step,
                                            energy_type = False,)
            elif self.input_type_x == 'images' and self.input_type_y == '1d' :
                plot_energy_image_1d_regression(self,
                                                save_dir=save_dir,
                                                samples_x = self.example_x,
                                                samples_y = [self.example_y,self.example_proposal_y],
                                                samples_title=['Samples from dataset','Samples from proposal'],
                                                name='contour',
                                                step=self.global_step,
                                                transform_back=self.transform_back,)
                for ebm_function, ebm_name in zip(ebm_function_list, ebm_function_name):
                    plot_energy_image_1d_regression(self,
                                                    save_dir=save_dir,
                                                    energy_function=ebm_function,
                                                    samples_x = self.example_x,
                                                    samples_y = [self.example_y,self.example_proposal_y],
                                                    samples_title=['Samples from dataset','Samples from proposal'],
                                                    name=ebm_name,
                                                    step=self.global_step,
                                                    energy_type = False,
                                                    transform_back=self.transform_back,)

