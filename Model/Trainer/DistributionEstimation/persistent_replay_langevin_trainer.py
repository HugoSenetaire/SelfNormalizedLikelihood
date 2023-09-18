import logging
import os
import random

import numpy as np
import torch
import torchvision
import wandb

from .abstract_trainer import AbstractDistributionEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)





class PersistentReplayLangevin(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal. This is not controlled by the proposal loss of the abstract trainer
    simply because the ebm is not properly defined in this case.
    """

    def __init__(
        self,
        ebm,
        cfg,
        device,
        logger,
        complete_dataset=None,
    ):
        super().__init__(
            ebm=ebm,
            cfg=cfg,
            device=device,
            logger=logger,
            complete_dataset=complete_dataset,
        )

        self.sigma_data = cfg.train.sigma_data

        assert self.ebm.proposal is not None, "The proposal should not be None"


    def training_energy(self, x):
        f_theta_opt, explicit_bias_opt, base_dist_opt, proposal_opt = self.optimizers
        f_theta_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        if self.sigma_data>0:
            x_data = x + torch.randn_like(x) * self.sigma_data
        else :
            x_data = x

        x_sample, _ = self.replay_buffer.get(n_samples=self.num_samples_train)
        energy_data, dic_output = self.ebm.calculate_energy(x_data)
        energy_samples, dic = self.ebm.calculate_energy(x_sample)
        for key in dic.keys():
            dic_output[key.replace("data", "samples")] = dic[key]
        loss_total = self.grads_and_reg(loss_energy=torch.mean(energy_data),
                                        loss_samples=torch.mean(-energy_samples),
                                        x=x,
                                        x_gen=x_sample,
                                        energy_data=energy_data,
                                        energy_samples=energy_samples,)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", torch.mean(energy_data))
        self.log("train/loss_samples", -torch.mean(energy_samples))

        f_theta_opt.step()
        explicit_bias_opt.step()
        if self.train_base_dist:
            base_dist_opt.step()
        f_theta_opt.zero_grad()
        base_dist_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        return loss_total, dic_output
