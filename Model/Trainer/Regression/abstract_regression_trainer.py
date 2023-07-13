import pytorch_lightning as pl
import torch
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_1d_1d_regression, plot_energy_image_1d_regression
from ...Utils.proposal_loss import proposal_loss_regression_getter
import numpy as np
import os


class AbstractRegression(pl.LightningModule):
    '''
    Abstract Trainer for EBM for distribution estimation.
    This makes sure that we compare all the different training process and loss in the same way.
    To provide a new training, one needs to inherit this class and define the training_step function.
    The main different with the distribution estimation counterpart is that we need to consider the input x
    for every calculation.

    Attributes:
    ----------
        ebm (EBM): The energy based model to train
        cfg (OmegaConfig): Config file containing all the parameters
        complete_dataset (Dataset): One of the dataset to sample from for visualization
        sampler (Sampler): The sampler to use for the visualization of the samples
        transform_back (function): The function to use to transform the samples back to the original space
                                    (for example, if the image is normalized, we need to unnormalize it)
        num_samples_train_estimate(int): Number of samples used to estimate the normalization constant 
                                        in training but not actually used in training. This is just for comparison
        num_samples_train (int): Number of samples used to estimate the normalization constant in training
        num_samples_val (int): Number of samples used to estimate the normalization constant in validation
        num_samples_test (int): Number of samples used to estimate the normalization constant in testing
        input_type (str): The type of input (1d, 2d, image, other)
        proposal_loss_name (str): The name of the loss used to train the proposal
        proposal_loss (function): The function used to train the proposal (Note that this function must be defined in the class)
        last_save (int): The last time the energy contour plot was saved

    Methods:
    -------
        training_step: The training step to be defined in inherited classes
        post_train_step_handler: Function called at the end of a train_step
        validation_step: The validation step
        validation_epoch_end: The validation epoch end
        test_step: The test step
        test_epoch_end: The test epoch end
        resample_base_dist: Resample the base distribution directly
        resample_proposal: Resample the proposal distribution directly
        proposal_visualization: Visualize the proposal distribution (if it exists)
        base_dist_visualization: Visualize the base distribution (if it exists)

    '''
    def __init__(self, ebm, cfg, complete_dataset = None,):
        super().__init__()
        self.ebm = ebm
        self.cfg = cfg
        self.last_save = -float('inf')

        self.nb_sample_train_estimate = cfg.proposal_training.num_sample_train_estimate
        self.num_samples_train = cfg.proposal_training.num_sample_proposal
        self.num_samples_val = cfg.proposal_training.num_sample_proposal_val
        self.num_samples_test = cfg.proposal_training.num_sample_proposal_test




        if np.prod(self.cfg.dataset.input_size_x) == 1 :
            self.input_type_x = '1d'
        elif len(self.cfg.dataset.input_size_x) == 3 :
            self.input_type_x = 'images'
        else :
            self.input_type_x = 'other'

        if np.prod(self.cfg.dataset.input_size_y) == 1 :
            self.input_type_y = '1d'
        elif np.prod(self.cfg.dataset.input_size_y) == 2 :
            self.input_type_y = '2d'
        else :
            self.input_type_y = 'other'

        if hasattr( complete_dataset, 'transform_back',):
            self.transform_back = complete_dataset.transform_back
        else :
            self.transform_back = None


        self.initialize_examples(complete_dataset)
        self.proposal_visualization()

        self.train_proposal = self.cfg.proposal_training.train_proposal

        for param in self.ebm.proposal.parameters():
            param.requires_grad = self.train_proposal


        self.proposal_loss_name = self.cfg.proposal_training.proposal_loss_name
        self.proposal_loss = proposal_loss_regression_getter(self.proposal_loss_name)
        self.test_name = None
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        '''
        The training step to be defined in inherited classes
        ''' 
        raise NotImplementedError
    
    def proposal_step(self, log_prob_proposal_data, estimate_log_z, proposal_opt, dic_output,):
        '''
        The proposal step. Calculates the loss of the proposal and updates the parameters of the proposal.

        Parameters:
        -----------
            log_prob_proposal_data (torch.Tensor): The log-probability of the data under the proposal
            estimate_log_z (torch.Tensor): The estimate of the log-normalization constant
            proposal_opt (torch.optim): The optimizer of the proposal
            dic_output (dict): The dictionary containing all the outputs to be handled.
        '''
        if self.train_proposal :
            proposal_opt.zero_grad()
            proposal_loss = self.proposal_loss(log_prob_proposal=log_prob_proposal_data, log_estimate_z=estimate_log_z)
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            self.log('train_proposal_loss', proposal_loss.mean())
            dic_output.update({"proposal_loss" : proposal_loss.mean()})
            proposal_opt.step()

    def validation_step(self, batch, batch_idx, name = 'val'):
        '''
        The validation step. Calculates the energy of the data and the log-normalization constant
        and l_SNL and l_IW.
        Also used for testubg gebce a specific name for specific proposal evaluation.

        Parameters:
        -----------
            batch (dict): The batch of data
            batch_idx (int): The index of the batch
            name (str): The name 'val' or 'test' or a specific name 

        Returns:
        --------
            dic_output (dict): The dictionary containing all the outputs to be handled.
        '''
        x = batch['data']
        y = batch['target']
        if name == 'val':
            num_samples = self.num_samples_val
        elif name == 'test':
            num_samples = self.num_samples_test
        else :
            raise NotImplementedError
        x_feature = self.ebm.feature_extractor(x)
        energy_data, dic_output = self.ebm.calculate_energy(x,y, x_feature=x_feature)
        energy_data = energy_data.reshape(x.shape[0],)
        
        log_prob_proposal_data = self.ebm.log_prob_proposal(x,y, x_feature=x_feature).reshape(x.shape[0],)
        save_dir = self.cfg.train.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        estimate_log_z, dic=self.ebm.estimate_log_z(x, nb_sample = num_samples, x_feature=x_feature)
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
        '''
        Proposal might be trained along with the EBM. In that case, we need to resample the proposal
        to visualize it.
        '''
        if self.ebm.proposal is not None :
            if self.input_type_y != 'other':
                if self.input_type_x == 'images':
                    self.example_proposal_x = self.example_x_default.to(dtype=self.dtype, device=self.device)
                    self.example_proposal_y = self.ebm.sample_proposal(self.example_proposal_x, 100).reshape(-1, 100, np.prod(self.cfg.dataset.input_size_y)).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_x = self.example_proposal_x.unsqueeze(1).expand(-1, 100, *self.cfg.dataset.input_size_x).reshape(-1, 100, *self.cfg.dataset.input_size_x).to(dtype=self.dtype)
                    self.min_y, self.max_y = min(torch.min(self.example_proposal_y), self.min_y_original), max(torch.max(self.example_proposal_y), self.max_y_original)
                elif self.input_type_x == '1d' :
                    self.example_proposal_x = torch.arange(self.min_x_original, self.max_x_original, (self.max_x_original-self.min_x_original)/100).reshape(-1, 1).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_y = self.ebm.sample_proposal(self.example_proposal_x, 100).reshape(-1, 100, np.prod(self.cfg.dataset.input_size_y)).to(dtype=self.dtype, device = self.device)
                    self.example_proposal_x = self.example_proposal_x.unsqueeze(1).expand(-1, 100, -1).reshape(-1, 100, 1)
                    self.min_y, self.max_y = min(torch.min(self.example_proposal_y), self.min_y_original), max(torch.max(self.example_proposal_y), self.max_y_original)
                else :
                    return None
    
    def initialize_examples(self, complete_dataset):
        '''
        Initialize the examples from the dataset to visualize along the energy contour plot.
        '''
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
            self.example_x = complete_dataset.dataset_train.__getitem__(indexes_to_print)['data'].reshape(-1, *self.cfg.dataset.input_size_x).to(dtype=self.dtype)
            self.example_x_default = self.example_x.clone()
            self.example_x = self.example_x.unsqueeze(1).expand(-1, 100, *self.cfg.dataset.input_size_x)
            self.example_y =complete_dataset.dataset_train.__getitem__(indexes_to_print)['target'].reshape(-1, np.prod(self.cfg.dataset.input_size_y)).unsqueeze(1).expand(-1, 100, *self.cfg.dataset.input_size_y)
            self.example_y = self.example_y.to(dtype=self.dtype)
            if self.input_type_x == '1d' :
                self.min_x_original, self.max_x_original = min(torch.min(self.example_x), self.min_x_original), max(torch.max(self.example_x), self.max_x_original)
            if self.input_type_y == '1d' or self.input_type_y == '2d':
                self.min_y_original, self.max_y_original = min(torch.min(self.example_y), self.min_y_original), max(torch.max(self.example_y), self.max_y_original)
        else :
            self.example_x, self.example_y = None, None




    def proposal_visualization(self, step = ''):
        '''
        Visualize the proposal distribution (if it exists) with energy and samples. 
        Samples will be saved in the experiment directory 'save_dir' in a folder 'proposal'
        depending on the steps values.

        Parameters:
        -----------
            step (int): The current step of the training
        '''

        if self.ebm.proposal is not None :
            self.resample_proposal()
            def current_energy_function(x,y):
                x_feature = self.ebm.feature_extractor(x)
                return -self.ebm.log_prob_proposal(x,y, x_feature=x_feature)
            # energy_function = lambda x,y: -self.ebm.log_prob_proposal(x,y)
            save_dir = self.cfg.train.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if np.prod(self.cfg.dataset.input_size_x) == 1 and np.prod(self.cfg.dataset.input_size_y) == 1:
                plot_energy_1d_1d_regression(self,
                                            save_dir=save_dir,
                                            name='proposal',
                                            energy_function=current_energy_function,
                                            samples_x=[self.example_x, self.example_proposal_x],
                                            samples_y=[self.example_y, self.example_proposal_y],
                                            samples_title=['data', 'proposal'],
                                            step=step,

                                            )
            elif self.input_type_x == 'images' and self.input_type_y == '1d':
                plot_energy_image_1d_regression(self,
                                                save_dir=save_dir,
                                                name='proposal',
                                                energy_function=current_energy_function,
                                                samples_x=self.example_x,
                                                samples_y=[self.example_y, self.example_proposal_y],
                                                samples_title=['data', 'proposal'],
                                                step=step,
                                                transform_back=self.transform_back,
                                                )
        
    def configure_optimizers(self):
        '''
        Configure the optimizers and schedulers for the training.
        
        Returns:
        --------
            optimizers (list): The list of optimizers, one for the EBM and one for the proposal
            schedulers (list): The list of schedulers if there are any
        '''
        parameters_ebm = [child.parameters() for name,child in self.ebm.named_children() if name != 'proposal']
        parameters_proposal = [self.ebm.proposal.parameters()] if self.ebm.proposal is not None else []
        
        ebm_opt = get_optimizer( cfg = self.cfg, list_parameters_gen = parameters_ebm)
        proposal_opt = get_optimizer( cfg = self.cfg, list_parameters_gen = parameters_proposal)

        ebm_sch = get_scheduler(cfg = self.cfg, optim = ebm_opt)
        proposal_sch = get_scheduler(cfg = self.cfg, optim = proposal_opt)
        if ebm_sch is not None and proposal_sch is not None :
            return [ebm_opt, proposal_opt], [ebm_sch, proposal_sch]      
        elif ebm_sch is not None :
            return [ebm_opt, proposal_opt], ebm_sch
        elif proposal_sch is not None :
            return [ebm_opt, proposal_opt], proposal_sch
        else :
            return [ebm_opt, proposal_opt]

        
    def update_dic_logger(self, outputs, name = 'val_'):
        '''
        Update the logger with the different outputs of the validation step or the test steps.
        '''
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
        '''
        Save the outputs of the validation step and update the logger.
        If possible, plot the energy contour and the proposals.
        '''
        self.update_dic_logger(outputs, name = 'val_')
        self.plot_energy()
        if self.train_proposal :
            self.resample_proposal()
            self.proposal_visualization(step = self.global_step)


        
       
    def test_step(self, batch, batch_idx):
        '''
        The test step. Calculates the energy of the data and the log-normalization constant
        and l_SNL and l_IW.Using the validation step implementation.
        '''
        if self.test_name is None :
            return self.validation_step(batch, batch_idx, name = 'test') # Regular testing
        else :
            return self.validation_step(batch, batch_idx, name = self.test_name) # Testing with a specific name for specific proposal evaluation
        
        
    def test_epoch_end(self, outputs):
        '''
        Save the outputs of the test step and update the logger.
        '''
        if self.test_name is None :
            self.update_dic_logger(outputs, name = 'test_') # Regular testing
        else :
            self.update_dic_logger(outputs, name = self.test_name+'_') # Testing with a specific name for specific proposal evaluation
        
    
    def plot_energy(self, ):
        '''
        Plot the energy contour if enough steps have passed since the last plot.
        Depending on the type of data points, the energy contour will be plotted differently.
        '''
        if self.global_step - self.last_save > self.cfg.train.save_energy_every :
            self.last_save = self.global_step
            self.resample_proposal()
            save_dir = self.cfg.train.save_dir
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

