import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from ...Sampler import get_sampler
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from .abstract_trainer import AbstractDistributionEstimation


class SelfNormalizedTrainer(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal.
    """

    def __init__(
        self,
        ebm,
        args_dict,
        complete_dataset=None,
        nb_sample_train_estimate=1024,
        **kwargs,
    ):
        super().__init__(
            ebm=ebm,
            args_dict=args_dict,
            complete_data=complete_dataset,
            nb_sample_train_estimate=nb_sample_train_estimate,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()

        if (
            self.args_dict["switch_mode"] is not None
            and self.global_step == self.args_dict["switch_mode"]
        ):
            self.ebm.switch_mode()
        x = batch["data"]
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)
        energy_samples, dic_output = self.ebm.calculate_energy(x)

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        estimate_log_z = estimate_log_z.mean()
        loss_estimate_z = estimate_log_z

        loss_energy = energy_samples.mean()
        loss_total = loss_energy + loss_estimate_z
        self.log("train_loss", loss_total)

        # Backward ebmxx
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total,
            retain_graph=True,
        )

        # Update the parameters of the proposal
        if self.train_proposal:
            proposal_opt.zero_grad()
            log_prob_proposal = self.ebm.proposal.log_prob(
                x,
            )
            self.log("proposal_log_likelihood", log_prob_proposal.mean())
            proposal_loss = self.proposal_loss(
                log_prob_proposal,
                estimate_log_z,
            )
            self.manual_backward(
                (proposal_loss).mean(), inputs=list(self.ebm.proposal.parameters())
            )
            proposal_opt.step()
        # Update the parameters of the ebm
        ebm_opt.step()
        dic_output.update(dic)

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
