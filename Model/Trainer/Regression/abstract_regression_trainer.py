import pytorch_lightning as pl
import torch
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_1d_regression
from ...Sampler import get_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml


class AbstractRegression(pl.LightningModule):
    def __init__(self, ebm, args_dict, complete_dataset = None, **kwargs):
        super().__init__()
        self.ebm = ebm
        self.args_dict = args_dict
        self.hparams.update(args_dict)
        self.last_save = -float('inf')
        # self.last_save_sample = -float('inf')
        self.last_save_sample = 0
        self.initialize_examples(complete_dataset)
        self.proposal_visualization()

        # print(self.args_dict['save_energy_every'])
        # assert False



    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    

    def validation_step(self, batch, batch_idx,):
        raise NotImplementedError
    

    
    def initialize_examples(self, complete_dataset):
        if np.prod(self.args_dict['input_size_x'])==1 and np.prod(self.args_dict['input_size_y'])==1 :
            if complete_dataset is not None :
                indexes_to_print = np.random.choice(len(complete_dataset.dataset_train), min(10000, len(complete_dataset.dataset_train)), replace=False)
                self.example_x = torch.cat([complete_dataset.dataset_train.__getitem__(i)[0] for i in indexes_to_print], dim=0)
                self.example_y = torch.cat([complete_dataset.dataset_train.__getitem__(i)[1].unsqueeze(0) for i in indexes_to_print], dim=0)
                self.min_x, self.max_x = min(torch.min(self.example_x,), -3), max(torch.max(self.example_x), 3)
                self.min_y, self.max_y = min(torch.min(self.example_y,), -3), max(torch.max(self.example_y), 3)
            else :
                self.example_x, self.example_y = None, None
                self.min_x, self.max_x, self.min_y, self.max_y = -3, 3, -3, 3
            if self.ebm.proposal is not None :
                self.example_proposal_x = torch.arange(self.min_x, self.max_x, 0.1).unsqueeze(1)
                batch_size = self.example_proposal_x.shape[0]
                self.example_proposal_y = self.ebm.proposal.sample(self.example_proposal_x, 100).reshape(-1, 1)
                self.example_proposal_x = self.example_proposal_x.unsqueeze(1).expand(batch_size, 100, 1).reshape(-1, 1)
                self.min_x, self.max_x = min(torch.min(self.example_proposal_x,), self.min_x), max(torch.max(self.example_proposal_x), self.max_x)
                self.min_y, self.max_y = min(torch.min(self.example_proposal_y,), self.min_y), max(torch.max(self.example_proposal_y), self.max_y)
            else :
                self.example_proposal = None
        elif len(self.args_dict['input_size_x'])==3 and np.prod(self.args_dict['input_size_y'])<3 :
            raise NotImplementedError
            # if complete_dataset is not None :
            #     self.example_x = torch.cat([complete_dataset.dataset_train.__getitem__(i)[0] for i in range(64)], dim=0)

            #     plot_images(self.example, save_dir=self.args_dict['save_dir'], name='example', transform_back=self.transform_back)
            # else :
            #     self.example = None
            
            # if self.ebm.proposal is not None :
            #     self.example_proposal = self.ebm.proposal.sample(64)
            #     plot_images(self.example_proposal, save_dir=self.args_dict['save_dir'], name='example_proposal', transform_back=self.transform_back)
            # else :
            #     self.example_proposal = None

    def proposal_visualization(self,):
        if self.ebm.proposal is not None :

            if np.prod(self.args_dict['input_size_x']) == 1 and np.prod(self.args_dict['input_size_y']) == 1:
                energy_function = lambda x,y: -self.ebm.proposal.log_prob(x,y)
                save_dir = self.args_dict['save_dir']
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_energy_1d_regression(self,
                                            save_dir=save_dir,
                                            name='proposal',
                                            energy_function=energy_function,
                                            samples_x=[self.example_x, self.example_proposal_x],
                                            samples_y=[self.example_y, self.example_proposal_y],
                                            samples_title=['data', 'proposal'],
                                            )
        
    def configure_optimizers(self):
        res = {}
        res['optimizer'] = get_optimizer(model = self.ebm, args_dict = self.args_dict)
        scheduler = get_scheduler(args_dict = self.args_dict, model = self.ebm, optim = res['optimizer'])
        if scheduler is not None:
            res['lr_scheduler'] = scheduler
        return res        

        
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


        
       
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        
        
    def test_epoch_end(self, outputs):
       self.update_dic_logger(outputs, name = 'test_')
        


    
    def plot_energy(self, ):
        if np.prod(self.args_dict['input_size_x']) == 1 and np.prod(self.args_dict['input_size_y']) == 1:
            if self.global_step - self.last_save > self.args_dict['save_energy_every'] :
                save_dir = self.args_dict['save_dir']
                save_dir = os.path.join(save_dir, "contour_energy")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_energy_1d_regression(self,
                                        save_dir=save_dir,
                                        samples_x = [self.example_x,self.example_proposal_x],
                                        samples_y = [self.example_y,self.example_proposal_y],
                                        samples_title=['Samples from dataset','Samples from proposal'],
                                        name='contour',
                                        step=self.global_step)
                ebm_function_list = [lambda x,y: self.ebm.calculate_energy(x,y)[1]['f_theta'],
                                    lambda x,y: self.ebm.calculate_energy(x,y)[1]['b'],]
                ebm_function_name = ['f_theta','b']
                
                for ebm_function, ebm_name in zip(ebm_function_list, ebm_function_name):
                    plot_energy_1d_regression(self,
                                            save_dir=save_dir,
                                            energy_function=ebm_function,
                                            samples_x = [self.example_x,self.example_proposal_x],
                                            samples_y = [self.example_y,self.example_proposal_y],
                                            samples_title=['Samples from dataset','Samples from proposal'],
                                            name=ebm_name,
                                            step=self.global_step,
                                            energy_type = False,)
                self.last_save = self.global_step

