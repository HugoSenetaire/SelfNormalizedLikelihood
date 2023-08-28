import itertools
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from omegaconf import OmegaConf, open_dict
from wandb import log, watch

from ...Sampler import get_sampler
from ...Utils.noise_annealing import calculate_current_noise_annealing
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from ...Utils.proposal_loss import proposal_loss_getter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class AbstractDistributionEstimation:
    """
    Abstract Trainer for EBM for distribution estimation.
    This makes sure that we compare all the different training loss in the same way.
    To provide a new training loss, one needs to inherit this class and define the training_step function.

    Attributes:
    ----------
        ebm (EBM): The energy based model to train
        cfg (dataclass): The dataclass containing all the arguments
        complete_dataset (Dataset): One of the dataset to sample from for visualization
        sampler (Sampler): The sampler to use for the visualization of the samples
        transform_back (function): The function to use to transform the samples back to the original space
                                    (for example, if the image is normalized, we need to unnormalize it)
        num_samples_train (int): Number of samples used to estimate the normalization constant in training
        num_samples_val (int): Number of samples used to estimate the normalization constant in validation
        num_samples_test (int): Number of samples used to estimate the normalization constant in test
        input_type (str): The type of input (1d, 2d, image, other)
        proposal_loss_name (str): The name of the loss used to train the proposal
        proposal_loss (function): The function used to train the proposal (Note that this function must be defined in the class)

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

    """

    def __init__(
        self,
        ebm,
        cfg,
        device,
        logger,
        dtype = torch.float32,
        complete_dataset=None,
    ):
        """
        Create a trainer using the EBM and the arguments dictionary.
        Verify the type of dataset used.
        Samples the base distribution and the proposal distribution for visualization.
        Populate some samples from dataset to save.

        Parameters:
        ----------
            ebm (EBM): The energy based model to train
            cfg (dataclass): The dataclass of arguments
            complete_dataset (Dataset): One of the dataset to sample from for visualization
            nb_sample_train_estimate (int): Number of samples to use for the estimation of the normalization constant to calculate the training loss
                                            This is not the same as num_samples_train which is effectively used during training and backprop,
                                            here it's only for providing a fair comparison between different number of samples.

        """
        super().__init__()
        self.ebm = ebm
        self.cfg = cfg
        self.dtype = dtype
        self.logger = logger
        self.sampler = get_sampler(cfg,)
        self.device = device
        self.current_step = 0
        
        if hasattr(complete_dataset, "transform_back"):
            self.transform_back = complete_dataset.transform_back
        else:
            self.transform_back = None

        self.nb_sample_train_estimate = cfg.proposal_training.num_sample_train_estimate
        self.num_samples_train = cfg.proposal_training.num_sample_proposal
        self.num_samples_val = cfg.proposal_training.num_sample_proposal_val
        self.num_samples_test = cfg.proposal_training.num_sample_proposal_test

        if np.prod(cfg.dataset.input_size) == 2:
            self.input_type = "2d"
        elif len(cfg.dataset.input_size) == 1:
            self.input_type = "1d"
        elif len(cfg.dataset.input_size) == 3:
            self.input_type = "image"
        else:
            self.input_type = "other"

        self.proposal_loss_name = cfg.proposal_training.proposal_loss_name
        self.proposal_loss = proposal_loss_getter(self.proposal_loss_name)

        self.train_proposal = cfg.proposal_training.train_proposal
        self.train_base_dist = cfg.base_distribution.train_base_dist
        if self.ebm.base_dist == self.ebm.proposal and self.train_proposal :
            self.train_base_dist = False

        self.configure_optimizers()
        self.initialize_examples(complete_dataset=complete_dataset)
        self.resample_base_dist()
        self.resample_proposal()
        self.proposal_visualization()
        self.base_dist_visualization()


    def on_train_start(self):
        watch(self.ebm.f_theta, log="all", log_freq=self.cfg.train.log_every_n_steps)
        watch(self.ebm.proposal, log="all", log_freq=self.cfg.train.log_every_n_steps)
        watch(self.ebm.base_dist, log="all", log_freq=self.cfg.train.log_every_n_steps)

    def log(self, name, value):
        self.logger.log({name :value}, step=self.current_step)

    def training_energy(self, x):
        """
        The training step to be defined in inherited classes.
        """
        raise NotImplementedError

    def base_dist_step(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        The training step to be defined in inherited classes.
        """
        x = batch["data"].to(self.device,)
        # print(self.train_proposal)
        # assert False
        if (self.train_proposal and self.cfg.train.nb_energy_steps is not None and self.cfg.train.nb_energy_steps > 0):
            if self.global_step % (self.cfg.train.nb_energy_steps + 1) != 0:
                loss, dic_output = self.training_energy(x,)
            else:
                loss, dic_output = self.proposal_step(x,)
        else:
            loss, dic_output = self.training_energy(x)
            if self.train_proposal:
                loss, dic_output_proposal = self.proposal_step(x)
                dic_output.update(dic_output_proposal)

        self.post_train_step_handler(x, dic_output,)

    def proposal_step(self,x,):
        """
        Update the parameters of the proposal to minimize the proposal loss.

        Parameters:
        ----------
            x (torch.Tensor): Batch
            estimate_log_z (torch.Tensor): The estimate of the log normalization constant
            proposal_opt (torch.optim): The optimizer for the proposal parameters
        """
        self.configure_gradient_flow("proposal")
        energy_opt, base_dist_opt, proposal_opt = self.optimizers
        proposal_opt.zero_grad()
        base_dist_opt.zero_grad()
        energy_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(
            x,
            self.num_samples_train,
            detach_sample=True,
            detach_base_dist=True,
        )

        current_noise = calculate_current_noise_annealing(
            self.current_step,
            self.cfg.proposal_training.noise_annealing_init,
            self.cfg.proposal_training.noise_annealing_gamma,
        )

        noisy_data = x + torch.randn_like(x) * current_noise

        log_prob_proposal_data = self.ebm.proposal.log_prob(noisy_data)
        proposal_loss = self.proposal_loss(
            log_prob_proposal_data,
            estimate_log_z,
        )

        proposal_loss.mean().backward()
        # proposal_loss.backward()
        if self.cfg.optim_proposal.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.proposal.parameters(),
                max_norm=self.cfg.optim_proposal.clip_grad_norm,
            )

        proposal_opt.step()
        self.log("train_proposal/extra_noise", current_noise)
        self.log("train_proposal/proposal_log_likelihood", log_prob_proposal_data.mean())
        self.log("train_proposal/estimate_log_z", estimate_log_z.mean())
        self.log("train_proposal/proposal_loss", proposal_loss.mean())


        return proposal_loss.mean(), dic

    def train(self, nb_steps, loader_train, val_loader=None):
        with torch.no_grad():
            self.ebm.eval()
            if val_loader is not None:
                liste_val = []
                for batch in tqdm.tqdm(val_loader):
                    self.validation_step(batch, self.current_step, liste_val)
                self.on_validation_epoch_end(outputs=liste_val)
    
        nb_epochs = int(np.ceil(nb_steps / len(loader_train)))
        for epoch in range(nb_epochs):
            print(f"Epoch {epoch} / {nb_epochs}")
            self.ebm.train()
            for batch in tqdm.tqdm(loader_train):
                self.training_step(batch, self.current_step)
                self.current_step += 1
            with torch.no_grad():
                self.ebm.eval()
                if val_loader is not None:
                    liste_val = []
                    for batch in tqdm.tqdm(val_loader):
                        self.validation_step(batch, self.current_step, liste_val)
                    self.on_validation_epoch_end(outputs=liste_val)
            if self.current_step >= nb_steps:
                break
        

    def train_bias(self, x, nb_iteration,):
        self.fix_base_dist()
        self.fix_f_theta()
        self.fix_proposal()
        self.free_log_bias()

        f_theta_opt, log_bias_opt, base_dist_opt, proposal_opt = self.optimizers
        for i in range(nb_iteration):   
            log_bias_opt.zero_grad()
            loss_estimate_z, _ = self.ebm.estimate_log_z(x, self.num_samples_train)
            loss_estimate_z = loss_estimate_z.exp() - 1
            if torch.isnan(loss_estimate_z) or torch.isinf(loss_estimate_z):
                logger.info("Loss estimate z is nan or inf")
                break
            loss_estimate_z.mean().backward()
            log_bias_opt.step()
        log_bias_opt.zero_grad()


    def post_train_step_handler(self, x, dic_output):
        """
        Function called at the end of a train_step.
        If the proposal is adapative, i.e. depending on the batch, make sure the batch is deleted.
        Create a new estimate of the log likelihood using a fixed number of samples
        (useful when evaluating multiple samples number)
        Log the different training outputs of the EBM coming from the dictionary, dic_output.

        Parameters:
        ----------
            x (torch.Tensor):
            dic_output (dict): The dictionary of outputs from the EBM
        """
        self.current_step += 1
        for scheduler in self.schedulers_no_feedback:
            if scheduler is not None :
                scheduler.step()
        

        for name, optim in zip(self.liste_optimizer_name, self.optimizers):
            for group in optim.param_groups:
                self.log(f"optim/{name}_lr", group["lr"])
                break

        with torch.no_grad():
            # Just in case it's an adaptive proposal that requires x
            if hasattr(self.ebm.proposal, "set_x"):
                self.ebm.proposal.set_x(None)
            self.ebm.eval()

            # Add some estimates of the log likelihood with a fixed number of samples independent from num samples proposal
            if (self.nb_sample_train_estimate is not None and self.nb_sample_train_estimate > 0):
                estimate_log_z, _ = self.ebm.estimate_log_z(x, nb_sample=self.nb_sample_train_estimate)
                log_likelihood_SNL = -dic_output["energy_on_data"].mean() - estimate_log_z.exp() + 1
                log_likelihood_IS = -dic_output["energy_on_data"].mean() - estimate_log_z
                self.log(f"train_fixed_{self.nb_sample_train_estimate}/log_likelihood_SNL", log_likelihood_SNL)
                self.log(f"train_fixed_{self.nb_sample_train_estimate}/log_likelihood_IS", log_likelihood_IS)

            for key in dic_output:
                self.log(f"train/{key}_mean", dic_output[key].mean().item())

            self.ebm.train()

    def validation_step(self, batch, batch_idx, liste = None):
        """
        Validation step, just returns the logs from calculating the energy of the EBM.
        """
        x = batch["data"].to(self.device, self.dtype)
        with torch.no_grad():
            loss, dic_output = self.ebm.calculate_energy(x)
        if liste is not None :
            liste.append(dic_output)
        return dic_output
    
    def test(self, dataloaders):
        for dataloader in dataloaders:
            for batch in tqdm.tqdm(dataloader):
                self.test_step(batch, self.current_step)
            self.on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        """
        The test step is the same as the validation step.
        """
        return self.validation_step(batch, batch_idx, type="test_" + self.test_type + "/")



    def on_validation_epoch_end(self, outputs, name="val/"):
        """
        Gather the energy from the batches in the validation step.
        Update the dictionary of outputs from the EBM by evaluating once the normalization constant.
        Visualize proposal, base_dist, energy and samples if number of step is sufficient.
        """
        self.update_dic_logger(outputs, name=name)
        self.proposal_visualization()
        self.base_dist_visualization()
        self.plot_energy()
        self.plot_samples()
        self.validation_step_outputs = []

    def on_test_epoch_end(self,):
        """
        Gather the energy from the batches in the test step.
        Update the dictionary of outputs from the EBM by evaluating once the normalization constant.
        """
        outputs = self.test_step_outputs
        self.update_dic_logger(outputs, name="test_" + self.test_type + "/")
        self.test_step_outputs = []

    def resample_base_dist(self,):
        """Base dist might be trained and need to be resampled during training"""
        if hasattr(self.ebm.base_dist, "sample"):
            if self.input_type == "2d":
                self.example_base_dist = self.ebm.base_dist.sample(1000).flatten(1)
            elif self.input_type == "1d":
                self.example_base_dist = self.ebm.base_dist.sample(1000)
            elif self.input_type == "image":
                self.example_base_dist = self.ebm.base_dist.sample(64)
        else:
            self.example_base_dist = None

    def resample_proposal(self,):
        """Proposal might be changing and need to be resampled during training"""
        if self.ebm.proposal is not None:
            if self.input_type == "2d":
                self.example_proposal = self.ebm.proposal.sample(1000).flatten(1)
            elif self.input_type == "1d":
                self.example_proposal = self.ebm.proposal.sample(1000)
            elif self.input_type == "image":
                self.example_proposal = self.ebm.proposal.sample(64)
        else:
            self.example_proposal = None

    def initialize_examples(self, complete_dataset):
        """
        Initialize the examples used for the visualization of the energy function.
        Making sure we have appropriate range for visualization and evaluation.
        """
        self.min_x, self.max_x, self.min_y, self.max_y = -3, 3, -3, 3
        if complete_dataset is not None:
            if self.input_type == "2d":
                indexes_to_print = np.random.choice(len(complete_dataset.dataset_train), 10000)
                self.example = torch.cat([complete_dataset.dataset_train.__getitem__(i)["data"].reshape(1, -1) for i in indexes_to_print],dim=0,)
                self.min_x, self.max_x = min(torch.min(self.example[:, 0],),-3,), max(torch.max(self.example[:, 0]), 3)
                self.min_y, self.max_y = min(torch.min(self.example[:, 1],),-3,), max(torch.max(self.example[:, 1]), 3)

            elif self.input_type == "1d":
                indexes_to_print = np.random.choice(len(complete_dataset.dataset_train), 10000)
                self.example = torch.cat([complete_dataset.dataset_train.__getitem__(i)["data"].unsqueeze(0) for i in indexes_to_print],dim=0,)
                self.min_x, self.max_x = min(torch.min(self.example[:, 0],),-3,), max(torch.max(self.example[:, 0]), 3)

            elif self.input_type == "image":
                if complete_dataset is not None:
                    self.example = torch.cat([complete_dataset.dataset_train.__getitem__(i)["data"] for i in range(64)],dim=0,)
                    plot_images(
                        images=self.example,
                        algo=self,
                        save_dir=self.cfg.train.save_dir,
                        name="example",
                        transform_back=self.transform_back,
                        step = self.current_step,
                    )
        else:
            self.example = None

    def proposal_visualization(self):
        """Visualize the proposal distribution and associated density.
        Depending on the input type, the visualization is different.
        """
        energy_function = lambda x: -self.ebm.proposal.log_prob(x)
        if self.input_type == "2d":
            self.resample_proposal()
            save_dir = os.path.join(self.cfg.train.save_dir, "proposal")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_energy_2d(
                self,
                energy_function=energy_function,
                save_dir=save_dir,
                samples=[self.example, self.example_proposal],
                samples_title=["Samples from dataset", "Samples from proposal"],
                name="proposal",
                step=self.current_step,
            )
        elif self.input_type == "image":
            self.resample_proposal()
            save_dir = os.path.join(self.cfg.train.save_dir, "proposal")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_images(
                images=self.example_proposal,
                algo=self,
                save_dir=save_dir,
                name="proposal_samples",
                transform_back=self.transform_back,
                step=self.current_step,
            )

    def base_dist_visualization(self):
        """Visualize the base distribution and associated density.
        Depending on the input type, the visualization is different."""
        if self.ebm.base_dist is not None:
            energy_function = lambda x: -self.ebm.base_dist.log_prob(x)
            if self.input_type == "2d":
                self.resample_proposal()
                save_dir = os.path.join(self.cfg.train.save_dir, "base_dist")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_energy_2d(
                    self,
                    energy_function=energy_function,
                    save_dir=save_dir,
                    samples=[
                        self.example,
                        self.example_base_dist,
                        self.example_proposal,
                    ],
                    samples_title=[
                        "Samples from dataset",
                        "Samples from base_dist",
                        "Samples from proposal",
                    ],
                    name="base_dist",
                    step=self.current_step,
                )
            elif self.input_type == "image":
                self.resample_base_dist()
                save_dir = os.path.join(self.cfg.train.save_dir, "base_dist")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_images(
                    images=self.example_base_dist,
                    algo=self,
                    save_dir=save_dir,
                    name="base_dist_samples",
                    transform_back=self.transform_back,
                    step=self.current_step,
                )

    def fix_f_theta(self,):
        for param in self.ebm.f_theta.parameters():
            param.requires_grad = False

    def free_f_theta(self,):
        for param in self.ebm.f_theta.parameters():
            param.requires_grad = True

    def fix_log_bias(self,):
        for param in self.ebm.explicit_bias.parameters():
            param.requires_grad = False
    
    def free_log_bias(self,):
        for param in self.ebm.explicit_bias.parameters():
            param.requires_grad = True

    def fix_proposal(self,):
        for param in self.ebm.proposal.parameters():
            param.requires_grad = False

    def free_proposal(self,):
        for param in self.ebm.proposal.parameters():
            param.requires_grad = True

    def fix_base_dist(self,):
        for param in self.ebm.base_dist.parameters():
            param.requires_grad = False

    def free_base_dist(self,):
        for param in self.ebm.base_dist.parameters():
            param.requires_grad = True

    def configure_gradient_flow(self, task="energy"):
        if task == "energy":
            self.free_f_theta()
            if self.ebm.base_dist == self.ebm.proposal:
                if self.train_base_dist:
                    self.free_base_dist()
                else:
                    self.fix_base_dist()
                    self.fix_proposal()
            else:
                self.fix_proposal()
                if self.train_base_dist:
                    self.free_base_dist()
                else:
                    self.fix_base_dist()
        elif task == 'bias':
            self.fix_proposal()
            self.fix_base_dist()
            self.free_f_theta()
        elif task == "proposal":
            self.fix_base_dist()
            self.fix_f_theta()
            self.free_proposal()
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        """
        Configure the optimizers for the EBM, base distribution and proposal.
        If the base distribution is equal to the proposal and the proposal is trained, the parameters ebm does not include the base distribution parameters.
        If the base distribution is equal to the proposal and the base distribution is trained, the proposal parameters are not given.
        It's not possible to train both

        Scheduler are treated differently depending on whether they require feedback from validation loss or not.
        They are stored in two different lists (self.schedulers_feedback and self.schedulers_no_feedback),
        if not scheduler is provided for either list, we just add none to the list.


        Returns:
        -------
            None
        """
        self.liste_optimizer_name = ['f_theta', 'log_bias', 'base_dist', 'proposal']
        self.cfg_scheduler = [self.cfg.scheduler_f_theta, self.cfg.scheduler_log_bias, self.cfg.scheduler_base_dist, self.cfg.scheduler_proposal]
        parameters_f_theta = [self.ebm.f_theta.parameters()]
        parameters_log_bias = [self.ebm.explicit_bias.parameters()]


        f_theta_opt = get_optimizer(cfg=self.cfg.optim_f_theta, list_parameters_gen=parameters_f_theta)
        log_bias_opt = get_optimizer(cfg=self.cfg.optim_log_bias, list_parameters_gen=parameters_log_bias)
        opt_list = [f_theta_opt, log_bias_opt]
        feedback_sch = []
        standard_sch = []
        get_scheduler(cfg=self.cfg.scheduler_f_theta, optim=f_theta_opt, feedback_scheduler=feedback_sch, standard_scheduler=standard_sch)
        get_scheduler(cfg=self.cfg.scheduler_log_bias, optim=log_bias_opt, feedback_scheduler=feedback_sch, standard_scheduler=standard_sch)


        # sch_list = [ebm_sch]

        if self.ebm.base_dist is not None:
            # print(self.ebm.base_dist.parameters())
            base_dist_opt = get_optimizer(cfg=self.cfg.optim_base_dist, list_parameters_gen=[self.ebm.base_dist.parameters()])
            get_scheduler(cfg=self.cfg.scheduler_base_dist, optim=base_dist_opt, feedback_scheduler=feedback_sch, standard_scheduler=standard_sch)
            opt_list.append(base_dist_opt)
        else :
            opt_list.append(None)
            feedback_sch.append(None)
            standard_sch.append(None)

        if self.ebm.proposal is not None:
            proposal_opt = get_optimizer(cfg=self.cfg.optim_proposal, list_parameters_gen=[self.ebm.proposal.parameters()])
            get_scheduler(cfg=self.cfg.scheduler_proposal, optim=proposal_opt, feedback_scheduler=feedback_sch, standard_scheduler=standard_sch)
            opt_list.append(proposal_opt)
        else :
            opt_list.append(None)
            feedback_sch.append(None)
            standard_sch.append(None)

        self.optimizers = opt_list
        self.schedulers_no_feedback = standard_sch
        self.schedulers_feedback = feedback_sch


    def update_dic_logger(self, outputs, name="val/"):
        """
        Update the dictionary of outputs from the EBM by evaluating once the normalization constant.
        Log the different training outputs of the EBM

        Parameters:
        ----------
            outputs (dict): The dictionary of saved logs obtained while calculating energy.
            name (str): The name of the outputs (val_ or test_)
        """
        list_keys = list(outputs[0].keys())
        dic_output = {}
        for key in list_keys:
            try :
                dic_output[name + key] = torch.cat([output[key] for output in outputs], dim=0)
            except RuntimeError:
                dic_output[name + key] = torch.cat([output[key].unsqueeze(0) for output in outputs], dim=0)
            self.log(name + key + "_mean", dic_output[name + key].mean())
            self.log(name + key + "_std", dic_output[name + key].std())

        with torch.no_grad():
            nb_sample = self.num_samples_val if name == "val/" else self.num_samples_test
            log_z_estimate, dic_output_estimate_z = self.ebm.estimate_log_z(
                self.example,
                nb_sample,
                detach_sample=True,
                detach_base_dist=True,
                return_samples=False,
            )
            for key in dic_output_estimate_z:
                self.log(name + key + "_mean", dic_output_estimate_z[key].mean())
                self.log(name + key + "_std", dic_output_estimate_z[key].std())
                # dic_output[name + key + "_mean"] = dic_output_estimate_z[key].mean()
                # dic_output[name + key + "_std"] = dic_output_estimate_z[key].std()



            energy_data = dic_output[name + "energy_on_data"].flatten()
            loss_total_SNL = - energy_data.mean() - log_z_estimate.exp() + 1
            self.log(name + "log_likelihood_SNL", loss_total_SNL)
            loss_total_IS = - energy_data.mean() - log_z_estimate.mean()
            self.log(name + "log_likelihood_IS", loss_total_IS)

        
        for scheduler, cfg in zip(self.schedulers_feedback, self.cfg_scheduler):
            if scheduler is not None:
                loss_for_scheduler = loss_total_SNL if cfg.feedback_loss == "SNL" else loss_total_IS
                scheduler.step(loss_for_scheduler)


    def gradient_control_l2(self, x, loss_energy, pg_control=0):
        """
        Add a gradient control term to the loss.
        """
        if pg_control == 0:
            return 0
        else:
            grad_e_data = (
                torch.autograd.grad(-loss_energy.sum(), x, create_graph=True)[0]
                .flatten(start_dim=1)
                .norm(2, 1)
            )
            return pg_control * (grad_e_data**2.0 / 2.0).mean()

    def plot_energy(
        self,
    ):
        """
        If possible show the current energy function and the samples from the proposal and dataset
        """
        if np.prod(self.cfg.dataset.input_size) == 2:
            if self.current_step % self.cfg.train.save_energy_every == 0 :
                save_dir = self.cfg.train.save_dir
                save_dir = os.path.join(save_dir, "contour_energy")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_energy_2d(
                    self,
                    save_dir=save_dir,
                    samples=[self.example, self.example_proposal],
                    samples_title=["Samples from dataset", "Samples from proposal"],
                    name="contour",
                    step=self.current_step,
                )

                # Add auxiliary contour plots, might be useful if we use a base distribution.
                ebm_function_list = [
                    lambda x,: self.ebm.calculate_energy(
                        x,
                    )[
                        1
                    ]["f_theta_on_data"],
                ]
                ebm_function_name = [
                    "f_theta",
                ]
                for ebm_function, ebm_name in zip(ebm_function_list, ebm_function_name):
                    plot_energy_2d(
                        self,
                        save_dir=save_dir,
                        energy_function=ebm_function,
                        samples=[self.example, self.example_proposal],
                        samples_title=["Samples from dataset", "Samples from proposal"],
                        name=ebm_name,
                        step=self.current_step,
                        energy_type=False,
                    )

    def samples_mcmc(self, num_samples=None):
        """
        Sample from the EBM distribution using an MCMC sampler.
        """
        if self.sampler is not None:
            samples, x_init = self.sampler.sample(
                self.ebm, self.ebm.proposal, num_samples=num_samples
            )
        else:
            return False
        return samples, x_init

    def plot_samples(self, num_samples=None):
        """
        Plot the samples from the EBM distribution using an MCMC sampler.
        """
        print("PLOTTING SAMPLES")
        if self.sampler is None:
            return False
        torch.set_grad_enabled(True)  # Required for MCMC sampling


        if self.current_step % self.cfg.train.samples_every == 0:
            save_dir = self.cfg.train.save_dir
            save_dir = os.path.join(save_dir, "samples_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            samples, init_samples = self.samples_mcmc(num_samples=num_samples)

            if self.input_type == "2d":
                samples = samples.flatten(1)
                plot_energy_2d(
                    self,
                    save_dir=save_dir,
                    samples=[samples],
                    samples_title=["HMC samples"],
                    name="samples",
                    step=self.current_step,
                )
            elif self.input_type == "image":
                print("Plotting images")
                plot_images(
                    algo=self,
                    save_dir=save_dir,
                    images=samples,
                    name="samples",
                    step=self.current_step,
                    init_samples=init_samples,
                    transform_back=self.transform_back,
                )
            else:
                raise NotImplementedError
