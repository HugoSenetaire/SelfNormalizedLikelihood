import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class NCETrainer(AbstractDistributionEstimation):
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

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()

        if (
            self.args_dict["switch_mode"] is not None
            and self.global_step == self.args_dict["switch_mode"]
        ):
            self.ebm.switch_mode()
        x = batch["data"]
        batch_size = x.shape[0]
        nb_sample = self.ebm.nb_sample

        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        energy_data, dic_output = self.ebm.calculate_energy(
            x,
        )
        energy_data = energy_data.reshape(
            x.shape[0],
        )
        log_prob_proposal_data = self.ebm.proposal.log_prob(x).reshape(
            x.shape[0],
        )

        samples = self.ebm.sample(nb_sample).to(x.device, x.dtype)
        energy_samples = (
            self.ebm.energy(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
        )
        if self.ebm.bias_explicit:
            energy_samples = self.ebm.explicit_bias_module(energy_samples)
        dic_output["f_theta_samples"] = energy_samples

        if self.ebm.base_dist is not None:
            base_dist_log_prob = (
                self.ebm.base_dist.log_prob(samples)
                .view(samples.size(0), -1)
                .sum(1)
                .unsqueeze(1)
            )
        else:
            base_dist_log_prob = torch.zeros_like(energy_samples).detach()
        energy_samples = energy_samples - base_dist_log_prob
        log_prob_proposal_samples = (
            self.ebm.proposal.log_prob(samples)
            .reshape(samples.shape[0], -1)
            .sum(1)
            .unsqueeze(1)
        )
        estimate_log_z = energy_samples - log_prob_proposal_samples

        logp_x = -energy_data.reshape(batch_size, 1)  # logp(x)
        logq_x = log_prob_proposal_data.reshape(batch_size, 1)  # logq(x)
        logp_gen = -energy_samples.reshape(nb_sample, 1)  # logp(x̃)
        logq_gen = log_prob_proposal_samples.reshape(nb_sample, 1)  # logq(x̃)
        log_noise_ratio = torch.log(
            torch.tensor(
                nb_sample / batch_size,
                dtype=energy_data.dtype,
                device=energy_data.device,
            )
        )

        value_data = logp_x - torch.logsumexp(
            torch.cat([logp_x, logq_x + log_noise_ratio], dim=1), dim=1, keepdim=True
        )  # logp(x)/(logp(x) + logq(x))
        value_gen = (
            logq_gen
            + log_noise_ratio
            - torch.logsumexp(
                torch.cat([logp_gen, logq_gen + log_noise_ratio], dim=1),
                dim=1,
                keepdim=True,
            )
        )  # logq(x̃)/(logp(x̃) + logq(x̃))

        # log_noise_ratio = torch.log(torch.tensor(nb_sample/batch_size, dtype=energy_data.dtype, device=energy_data.device))

        # value_data = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
        # value_gen = logq_gen - torch.logsumexp(torch.cat([logp_gen, logq_gen], dim=1), dim=1, keepdim=True)  # logq(x̃)/(logp(x̃) + logq(x̃))
        nce_objective = value_data.mean() + (value_gen + log_noise_ratio).mean()
        nce_loss = -nce_objective

        loss_total = nce_loss

        # Backward ebm
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total,
            retain_graph=True,
        )
        self.log("train_loss", loss_total)

        # Update the parameters of the proposal
        if self.train_proposal:
            proposal_opt.zero_grad()
            self.log("proposal_log_likelihood", log_prob_proposal_data.mean())
            proposal_loss = self.proposal_loss(
                log_prob_proposal_data,
                estimate_log_z,
            )
            self.manual_backward(
                (proposal_loss).mean(), inputs=list(self.ebm.proposal.parameters())
            )
            proposal_opt.step()

        # Update the parameters of the ebm
        ebm_opt.step()
        # dic_output.update(dic)

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
