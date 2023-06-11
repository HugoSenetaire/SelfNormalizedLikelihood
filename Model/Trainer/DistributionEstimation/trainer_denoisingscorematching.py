import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.autograd
import yaml

from ...Sampler import get_sampler
from ...Utils.optimizer_getter import get_optimizer, get_scheduler
from ...Utils.plot_utils import plot_energy_2d, plot_images
from .abstract_trainer import AbstractDistributionEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class DenoisingScoreMatchingTrainer(AbstractDistributionEstimation):
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
        **kwargs
    ):
        logger.warning("args_dict is deprecated, use cfg instead")
        raise NotImplementedError
        super().__init__(
            ebm=ebm,
            args_dict=args_dict,
            complete_data=complete_dataset,
            nb_sample_train_estimate=nb_sample_train_estimate,
            **kwargs
        )
        if "sigma" in self.args_dict.keys():
            self.sigma = self.args_dict["sigma"]
        else:
            self.sigma = 0.1

    def denoising_score_matching(
        self,
        data,
    ):
        data = data.flatten(1)
        data.requires_grad_(True)
        vector = torch.randn_like(data) * self.sigma
        perturbed_data = data + vector
        energy_perturbed_data, dic = self.ebm.calculate_energy(perturbed_data)
        logp = -energy_perturbed_data  # logp(x)

        dlogp = (
            self.sigma**2
            * torch.autograd.grad(logp.sum(), perturbed_data, create_graph=True)[0]
        )
        kernel = vector
        loss = torch.norm(dlogp + kernel, dim=-1) ** 2
        loss = loss.mean() / 2.0

        return loss, dic

    def training_step(self, batch, batch_idx):
        ebm_opt, proposal_opt = self.optimizers_perso()

        if (
            self.args_dict["switch_mode"] is not None
            and self.global_step == self.args_dict["switch_mode"]
        ):
            self.ebm.switch_mode()
        x = batch["data"]
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        loss_total, dic_output = self.denoising_score_matching(x)
        self.log("train_loss", loss_total.mean())
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total.mean(),
            retain_graph=False,
        )
        ebm_opt.step()

        ebm_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.ebm.nb_sample)
        estimate_log_z = estimate_log_z.mean()
        dic_output.update(dic)

        # Update the parameters of the proposal
        if self.train_proposal:
            proposal_opt.zero_grad()
            log_prob_proposal_data = self.ebm.proposal.log_prob(
                x,
            )
            self.log("proposal_log_likelihood", log_prob_proposal_data.mean())
            proposal_loss = self.proposal_loss(
                log_prob_proposal_data,
                estimate_log_z,
            )
            self.manual_backward(
                (proposal_loss).mean(), inputs=list(self.ebm.proposal.parameters())
            )
            proposal_opt.step()

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
