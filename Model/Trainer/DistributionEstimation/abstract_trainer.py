import itertools
import logging
import os
import numpy as np
import pytorch_lightning as pl
import torch

from ...Sampler import get_sampler
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from ...Utils.proposal_loss import proposal_loss_getter
from ...Sampler import get_sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class AbstractDistributionEstimation(pl.LightningModule):
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

    """

    def __init__(
        self,
        ebm,
        cfg,
        complete_dataset=None,
    ):
        '''
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
                                            
        '''
        super().__init__()
        self.ebm = ebm
        self.cfg = cfg
        # self.hparams.update(args_dict)
        logger.info(f"You might want to save some hparams here")
        self.last_save = -float("inf")  # To save the energy contour plot
        self.last_save_sample = 0  # To save the samples
        self.sampler = get_sampler(cfg,)
        if hasattr(self.ebm, "transform_back"):
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
        self.initialize_examples(complete_dataset=complete_dataset)
        self.resample_base_dist()
        self.resample_proposal()
        self.proposal_visualization()
        self.base_dist_visualization()
        self.automatic_optimization = False
        self.train_proposal = cfg.proposal_training.train_proposal
        self.train_base_dist = cfg.base_distribution.train_base_dist

        if self.ebm.base_dist is not None :
            for param in self.ebm.base_dist.parameters():
                param.requires_grad = self.train_base_dist

        for param in self.ebm.proposal.parameters():
            param.requires_grad = self.train_proposal
        
        if self.ebm.base_dist == self.ebm.proposal:
            # Overwrite before if base dist == proposal and one of them is trained
            if self.train_proposal or self.train_base_dist:
                for param in self.ebm.proposal.parameters():
                    param.requires_grad = True

    def training_step(self, batch, batch_idx):
        """
        The training step to be defined in inherited classes.
        """
        raise NotImplementedError
    
    def _proposal_step(self, x, estimate_log_z, proposal_opt, dic_output):
        '''
        Update the parameters of the proposal to minimize the proposal loss.
        
        Parameters:
        ----------
            x (torch.Tensor): Batch
            estimate_log_z (torch.Tensor): The estimate of the log normalization constant
            proposal_opt (torch.optim): The optimizer for the proposal parameters
        '''
        if self.train_proposal :
            proposal_opt.zero_grad()
            log_prob_proposal_data = self.ebm.proposal.log_prob(x,)
            self.log('proposal_log_likelihood', log_prob_proposal_data.mean())
            proposal_loss = self.proposal_loss(log_prob_proposal_data, estimate_log_z,)
            dic_output.update({"proposal_loss" : proposal_loss.mean()})
            self.manual_backward((proposal_loss).mean(), inputs= list(self.ebm.proposal.parameters()))
            proposal_opt.step()

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
        # Just in case it's an adaptive proposal that requires x
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(None)

        # Add some estimates of the log likelihood with a fixed number of samples independent from num samples proposal
        if (self.nb_sample_train_estimate is not None and self.nb_sample_train_estimate > 0):
            estimate_log_z, _ = self.ebm.estimate_log_z(x, nb_sample=self.nb_sample_train_estimate)
            SNL_fix_z = -dic_output["energy"].mean() - estimate_log_z.exp() + 1
            log_likelihood = -dic_output["energy"].mean() - estimate_log_z
            dic_output[f"SNL_{self.nb_sample_train_estimate}"] = SNL_fix_z
            dic_output[
                f"log_likelihood_{self.nb_sample_train_estimate}"
            ] = log_likelihood

        for key in dic_output:
            self.log(f"train_{key}_mean", dic_output[key].mean().item())

    def validation_step(self, batch, batch_idx):
        """
        Validation step, just returns the logs from calculating the energy of the EBM.
        """
        x = batch["data"]
        energy_batch, dic_output = self.ebm.calculate_energy(x)
        return dic_output

    def test_step(self, batch, batch_idx):
        """
        The test step is the same as the validation step.
        """
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        """
        Gather the energy from the batches in the validation step.
        Update the dictionary of outputs from the EBM by evaluating once the normalization constant.
        Visualize proposal, base_dist, energy and samples if number of step is sufficient.
        """
        self.update_dic_logger(outputs, name="val_")
        self.proposal_visualization()
        self.base_dist_visualization()
        self.plot_energy()
        self.plot_samples()

    def test_epoch_end(self, outputs):
        """
        Gather the energy from the batches in the test step.
        Update the dictionary of outputs from the EBM by evaluating once the normalization constant.
        """
        self.update_dic_logger(outputs, name="test_")

    def resample_base_dist(
        self,
    ):
        """Base dist might be trained and need to be resampled during training"""
        if hasattr(self.ebm.base_dist, 'sample'):
            if self.input_type == "2d":
                self.example_base_dist = self.ebm.base_dist.sample(1000).flatten(1)
            elif self.input_type == "1d":
                self.example_base_dist = self.ebm.base_dist.sample(1000)
            elif self.input_type == "image":
                self.example_base_dist = self.ebm.base_dist.sample(64)
        else:
            self.example_base_dist = None

    def resample_proposal(
        self,
    ):
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
                indexes_to_print = np.random.choice(
                    len(complete_dataset.dataset_train), 10000
                )
                self.example = torch.cat(
                    [
                        complete_dataset.dataset_train.__getitem__(i)['data'].reshape(1, -1)
                        for i in indexes_to_print
                    ],
                    dim=0,
                )
                self.min_x, self.max_x = min(
                    torch.min(
                        self.example[:, 0],
                    ),
                    -3,
                ), max(torch.max(self.example[:, 0]), 3)
                self.min_y, self.max_y = min(
                    torch.min(
                        self.example[:, 1],
                    ),
                    -3,
                ), max(torch.max(self.example[:, 1]), 3)

            elif self.input_type == "1d":
                indexes_to_print = np.random.choice(
                    len(complete_dataset.dataset_train), 10000
                )
                self.example = torch.cat(
                    [
                        complete_dataset.dataset_train.__getitem__(i)['data'].unsqueeze(0)
                        for i in indexes_to_print
                    ],
                    dim=0,
                )
                self.min_x, self.max_x = min(
                    torch.min(
                        self.example[:, 0],
                    ),
                    -3,
                ), max(torch.max(self.example[:, 0]), 3)

            elif self.input_type == "image":
                if complete_dataset is not None:
                    self.example = torch.cat(
                        [
                            complete_dataset.dataset_train.__getitem__(i)['data']
                            for i in range(64)
                        ],
                        dim=0,
                    )
                    plot_images(
                        self.example,
                        save_dir=self.cfg.train.save_dir,
                        name="example",
                        transform_back=self.transform_back,
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
                step=self.global_step,
            )
        elif self.input_type == "image":
            self.resample_proposal()
            save_dir = os.path.join(self.cfg.train.save_dir, "proposal")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_images(
                self.example_proposal,
                save_dir=save_dir,
                name="proposal_samples",
                transform_back=self.transform_back,
                step=self.global_step,
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
                    step=self.global_step,
                )
            elif self.input_type == "image":
                self.resample_base_dist()
                save_dir = os.path.join(self.cfg.train.save_dir, "base_dist")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_images(
                    self.example_base_dist,
                    save_dir=save_dir,
                    name="base_dist_samples",
                    transform_back=self.transform_back,
                    step=self.global_step,
                )

    def configure_optimizers(self):
        """
        Configure the optimizers for the EBM, base distribution and proposal.
        If the base distribution is equal to the proposal and the proposal is trained, the parameters ebm does not include the base distribution parameters.
        If the base distribution is equal to the proposal and the base distribution is trained, the proposal parameters are not given.
        It's not possible to train both

        Returns:
        -------
            opt_list (list): The list of optimizers
            sch_list (list): The list of schedulers
        """
        if self.cfg.proposal_training.train_proposal and self.ebm.proposal == self.ebm.base_dist:
            # In case the base dist is equal to the proposal, I can't train both of them with the same loss
            # If I want to train the proposal it takes priority over the base distribution
            parameters_ebm = [
                child.parameters()
                for name, child in self.ebm.named_children()
                if name != "proposal" and name != "base_dist"
            ]
            print("Proposal takes priority here")
        else:
            parameters_ebm = [
                child.parameters()
                for name, child in self.ebm.named_children()
                if name != "proposal"
            ]
        # parameters_ebm.append(self.ebm.parameters())
        ebm_opt = get_optimizer(cfg=self.cfg, list_parameters_gen=parameters_ebm)
        ebm_sch = get_scheduler(cfg=self.cfg, optim=ebm_opt)

        if (
            not self.cfg.proposal_training.train_proposal
            and self.ebm.proposal == self.ebm.base_dist
        ):
            proposal_opt = None
            proposal_sch = None
        else:
            parameters_proposal = (
                [self.ebm.proposal.parameters()]
                if self.ebm.proposal is not None
                else []
            )
            proposal_opt = get_optimizer(
                cfg=self.cfg, list_parameters_gen=parameters_proposal
            )
            proposal_sch = get_scheduler(cfg=self.cfg, optim=proposal_opt)

        opt_list = [ebm_opt]
        if proposal_opt is not None:
            opt_list.append(proposal_opt)
        if ebm_sch is not None and proposal_sch is not None:
            return opt_list, [ebm_sch, proposal_sch]
        elif ebm_sch is not None:
            return opt_list, ebm_sch
        elif proposal_sch is not None:
            return opt_list, proposal_sch
        else:
            return opt_list

    def optimizers_perso(self):
        """
        I don't remember why I needed that. #TODO : CHECK WHY I WROTE THIS.
        """
        liste_opt = super().optimizers()
        try:
            ebm_opt, proposal_opt = liste_opt
            return liste_opt
        except:
            return liste_opt, None

    def update_dic_logger(self, outputs, name="val_"):
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
            try:
                dic_output[name + key + "_mean"] = torch.cat(
                    [outputs[k][key] for k in range(len(outputs))]
                ).mean()
            except RuntimeError:
                dic_output[name + key + "_mean"] = torch.cat(
                    [outputs[k][key].unsqueeze(0) for k in range(len(outputs))]
                ).mean()
        mean_energy = dic_output[name + "energy_mean"]

        nb_sample = self.num_samples_val if name == "val_" else self.num_samples_test
        log_z_estimate, dic = self.ebm.estimate_log_z(x=torch.zeros((1,), dtype=torch.float32, device=self.device), nb_sample=nb_sample,)


        dic_output.update({name + k + "_mean": v.mean() for k, v in dic.items()})
        total_loss_self_norm = mean_energy + log_z_estimate.exp()
        self.log(name + "loss_self_norm", total_loss_self_norm)
        total_likelihood = -mean_energy - log_z_estimate.exp() + 1
        self.log(name + "likelihood_normalized", total_likelihood)

        total_loss_self_norm = mean_energy + log_z_estimate
        self.log(name + "loss_log", total_loss_self_norm)
        total_likelihood = -mean_energy - log_z_estimate
        self.log(name + "likelihood_log", total_likelihood)

        self.log("val_loss", total_loss_self_norm)

        for key in dic_output:
            self.log(key, dic_output[key])

    def plot_energy(
        self,
    ):
        """
        If possible show the current energy function and the samples from the proposal and dataset
        """
        if np.prod(self.cfg.dataset.input_size) == 2:
            if self.global_step - self.last_save > self.cfg.train.save_energy_every:
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
                    step=self.global_step,
                )

                # Add auxiliary contour plots, might be useful if we use a base distribution.
                ebm_function_list = [
                    lambda x,: self.ebm.calculate_energy(
                        x,
                    )[
                        1
                    ]["f_theta"],
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
                        step=self.global_step,
                        energy_type=False,
                    )

                self.last_save = self.global_step

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
        if self.sampler is None:
            return False
        torch.set_grad_enabled(True)  # Required for MCMC sampling

        if self.global_step - self.last_save_sample > self.cfg.train.samples_every:
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
                    step=self.global_step,
                )
            elif self.input_type == "image":
                plot_images(
                    algo=self,
                    save_dir=save_dir,
                    images=samples,
                    name="samples",
                    step=self.global_step,
                    init_samples=init_samples,
                    transform_back=self.transform_back,
                )
            else:
                raise NotImplementedError
            self.last_save_sample = self.global_step
