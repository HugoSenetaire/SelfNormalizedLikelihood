import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from ...Sampler import get_sampler
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from ...Utils.proposal_loss import log_prob_kl_loss, kl_loss, log_prob_loss
from ...Sampler import get_sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import itertools



class AbstractDistributionEstimation(pl.LightningModule):
    def __init__(
        self,
        ebm,
        args_dict,
        complete_dataset=None,
        nb_sample_train_estimate=1024,
        **kwargs,
    ):
        super().__init__()
        self.ebm = ebm
        self.args_dict = args_dict
        print("args_dict", args_dict)
        self.hparams.update(args_dict)
        self.last_save = -float('inf') # To save the energy contour plot 
        self.last_save_sample = 0 # To save the samples
        self.sampler = get_sampler(args_dict,)
        self.transform_back = complete_dataset.transform_back
        self.nb_sample_train_estimate = nb_sample_train_estimate

        if np.prod(self.args_dict["input_size"]) == 2:
            self.input_type = "2d"
        elif len(self.args_dict["input_size"]) == 1:
            self.input_type = "1d"
        elif len(self.args_dict["input_size"]) == 3:
            self.input_type = "image"
        else:
            self.input_type = "other"

        self.proposal_loss_name = args_dict["proposal_loss_name"]
        if self.proposal_loss_name == "log_prob":
            self.proposal_loss = log_prob_loss
        elif self.proposal_loss_name == "kl":
            self.proposal_loss = kl_loss
        elif self.proposal_loss_name == "log_prob_kl":
            self.proposal_loss = log_prob_kl_loss
        else:
            raise ValueError("Proposal loss name not recognized")

        self.initialize_examples(complete_dataset=complete_dataset)
        self.proposal_visualization()
        self.automatic_optimization = False
        self.train_proposal = self.args_dict["train_proposal"]

        if not self.train_proposal:
            for param in self.ebm.proposal.parameters():
                param.requires_grad = False
        else:
            for param in self.ebm.proposal.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def post_train_step_handler(self, x, dic_output):
        """
        Function called at the end of a train_step
        """
        # Just in case it's an adaptive proposal that requires x
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(None)

        # Add some estimates of the log likelihood with a fixed number of samples
        if (
            self.nb_sample_train_estimate is not None
            and self.nb_sample_train_estimate > 0
        ):
            estimate_log_z, _ = self.ebm.estimate_log_z(
                x, nb_sample=self.nb_sample_train_estimate
            )
            log_likelihood_fix_z = -dic_output["energy"].mean() - estimate_log_z + 1
            self.log("train_log_likelihood_fix_z", log_likelihood_fix_z)
        for key in dic_output:
            self.log(f"train_{key}_mean", dic_output[key].mean().item())

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        energy_samples, dic_output = self.ebm.calculate_energy(x)
        return dic_output

    def validation_epoch_end(self, outputs):
        self.update_dic_logger(outputs, name="val_")
        self.proposal_visualization()
        self.plot_energy()
        self.plot_samples()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.update_dic_logger(outputs, name="test_")

    def resample_proposal(
        self,
    ):
        """Proposal might be changing and need to be resampled during training"""
        if self.ebm.proposal is not None:
            if self.input_type == "2d":
                self.example_proposal = self.ebm.proposal.sample(1000).flatten(1)
                self.min_x, self.max_x = min(
                    torch.min(
                        self.example_proposal[:, 0],
                    ),
                    self.min_x,
                ), max(torch.max(self.example_proposal[:, 0]), self.max_x)
                self.min_y, self.max_y = min(
                    torch.min(
                        self.example_proposal[:, 1],
                    ),
                    self.min_y,
                ), max(torch.max(self.example_proposal[:, 1]), self.max_y)
            elif self.input_type == "1d":
                self.example_proposal = self.ebm.proposal.sample(1000)
                self.min_x, self.max_x = min(
                    torch.min(self.example_proposal), self.min_x
                ), max(torch.max(self.example_proposal), self.max_x)
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
                        complete_dataset.dataset_train.__getitem__(i)[0]
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
                        complete_dataset.dataset_train.__getitem__(i)[0].unsqueeze(0)
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
                            complete_dataset.dataset_train.__getitem__(i)[0]
                            for i in range(64)
                        ],
                        dim=0,
                    )
                    plot_images(
                        self.example,
                        save_dir=self.args_dict["save_dir"],
                        name="example",
                        transform_back=self.transform_back,
                    )
        else:
            self.example = None

    def proposal_visualization(self):
        """Visualize the proposal"""
        energy_function = lambda x: -self.ebm.proposal.log_prob(x)
        if self.input_type == "2d":
            self.resample_proposal()
            save_dir = os.path.join(self.args_dict["save_dir"], "proposal")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_energy_2d(
                self,
                energy_function=energy_function,
                save_dir=save_dir,
                samples=[self.example, self.example_proposal],
                samples_title=["Samples from dataset", "Samples from proposal"],
                name="proposal",
            )
        elif self.input_type == "image":
            self.resample_proposal()
            save_dir = os.path.join(self.args_dict["save_dir"], "proposal")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_images(
                self.example_proposal,
                save_dir=save_dir,
                name="proposal_samples",
                transform_back=self.transform_back,
                step=self.global_step,
            )

    def configure_optimizers(self):
        params_ebm = [child.parameters() for name,child in self.ebm.named_children() if name != 'proposal']
        params_ebm.append(self.ebm.parameters())
        params_proposal = [self.ebm.proposal.parameters()] if self.ebm.proposal is not None else []
        ebm_opt = get_optimizer( args_dict = self.args_dict, list_params_gen = params_ebm)
        proposal_opt = get_optimizer( args_dict = self.args_dict, list_params_gen = params_proposal)

        ebm_sch = get_scheduler(args_dict = self.args_dict, optim = ebm_opt)
        proposal_sch = get_scheduler(args_dict = self.args_dict, optim = proposal_opt)
        if ebm_sch is not None and proposal_sch is not None :
            return [ebm_opt, proposal_opt], [ebm_sch, proposal_sch]      
        elif ebm_sch is not None :
            return [ebm_opt, proposal_opt], ebm_sch
        elif proposal_sch is not None:
            return [ebm_opt, proposal_opt], proposal_sch
        else:
            return [ebm_opt, proposal_opt]

    def update_dic_logger(self, outputs, name="val_"):
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
        # dic_output = {name+key+"_mean": torch.cat([outputs[k][key] for k in range(len(outputs))]).mean() for key in list_keys}
        mean_energy = dic_output[name + "energy_mean"]
        log_z_estimate, dic = self.ebm.estimate_log_z(
            x=torch.zeros((1,), dtype=torch.float32, device=self.device),
            nb_sample=self.args_dict["num_sample_proposal_test"],
        )

        dic_output.update({name + k + "_mean": v.mean() for k, v in dic.items()})
        total_loss_self_norm = mean_energy + log_z_estimate.exp()
        self.log(name + "loss_self_norm", total_loss_self_norm)
        total_likelihood = -mean_energy - log_z_estimate.exp() + 1
        self.log(name + "likelihood_normalized", total_likelihood)

        if self.ebm.type_z == "exp":
            self.log(name + "loss", total_loss_self_norm)

        total_loss_self_norm = mean_energy + log_z_estimate
        self.log(name + "loss_log", total_loss_self_norm)
        total_likelihood = -mean_energy - log_z_estimate
        self.log(name + "likelihood_log", total_likelihood)

        if self.ebm.type_z == "log":
            self.log(name + "loss", total_loss_self_norm)

        for key in dic_output:
            self.log(key, dic_output[key])

    def plot_energy(
        self,
    ):
        """
        If possible show the current energy function and the samples from the proposal and dataset
        """
        if np.prod(self.args_dict["input_size"]) == 2:
            if self.global_step - self.last_save > self.args_dict["save_energy_every"]:
                save_dir = self.args_dict["save_dir"]
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
        samples, x_init = self.sampler.sample(
            self.ebm, self.ebm.proposal, num_samples=num_samples
        )
        return samples, x_init

    def plot_samples(self, num_samples=None):
        """
        Plot the samples from the EBM distribution using the sampler
        """
        torch.set_grad_enabled(True)
        if self.global_step - self.last_save_sample > self.args_dict["samples_every"]:
            save_dir = self.args_dict["save_dir"]
            save_dir = os.path.join(save_dir, "samples_energy")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            samples, init_samples = self.samples_mcmc(num_samples=num_samples)

            if np.prod(self.args_dict["input_size"]) == 2:
                samples = samples.flatten(1)
                plot_energy_2d(
                    self,
                    save_dir=save_dir,
                    samples=[samples],
                    samples_title=["HMC samples"],
                    name="samples",
                    step=self.global_step,
                )
            # elif len(self.args_dict["input_size"]) == 2 and self.args_dict["input_size"][0]==1:
            #     plot_energy_1d()
            elif len(self.args_dict["input_size"]) == 3:
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
