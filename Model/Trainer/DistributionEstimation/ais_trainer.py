import logging
import os
import random

import numpy as np
import torch
import torchvision
import wandb

from .abstract_trainer import AbstractDistributionEstimation
from ...Utils.Buffer import AIS_Sample_Buffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)





class AnnealedImportanceSampling(AbstractDistributionEstimation):
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



        assert (self.proposal is not None or self.ebm.base_dist is not None), "The proposal or the base dist should not be None"
        assert not self.train_base_dist, "The base dist should not be trained"
        assert not self.train_proposal, "The proposal should not be trained"

        self.ais_sample_buffer = AIS_Sample_Buffer(cfg=cfg.train, buffer_name="ais_sample_buffer")
        self.ais_sample_buffer.populate_buffer(self.ebm)



    def training_energy(self, x):
        # Get parameters
        f_theta_opt, explicit_bias_opt, _, _, = self.optimizers
        f_theta_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        x_gen, weights_x_gen = self.ais_sample_buffer.get(self.num_samples_train, device=self.device)
        x_gen = x_gen.detach()
        energy_samples, dic = self.ebm.calculate_energy(x_gen.reshape(self.num_samples_train, *x.shape[1:]))
        
        energy_data, dic_output = self.ebm.calculate_energy(x)

        for key, value in dic.items():
            dic_output[key+"sample"] = value.mean()

        loss_sample = (-energy_samples.flatten()-weights_x_gen.flatten()).logsumexp(0)
        loss_sample = loss_sample - torch.log(torch.tensor(self.num_samples_train, dtype=torch.float, device=self.device))
        # loss_sample = loss_sample.exp()
        loss_energy = energy_data.mean()

        loss_total = self.backward_and_reg(loss_energy=loss_energy,
                        loss_samples=loss_sample,
                        x=x,
                        x_gen=x_gen,
                        energy_data=energy_data,
                        energy_samples=energy_samples,)
        self.grad_clipping()
        self.ais_sample_buffer.update_buffer(self.ebm)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_sample", loss_sample)

        f_theta_opt.step()
        explicit_bias_opt.step()
        f_theta_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        self.log("ais_buffer/buffer_weights_mean", torch.stack(self.ais_sample_buffer.buffer_weights).mean())
        self.log("ais_buffer/buffer_weights_std", torch.stack(self.ais_sample_buffer.buffer_weights).std())
        if self.current_step % self.cfg.train.save_buffer_every == 0:
            self.ais_sample_buffer.save_buffer(self)
            

        return loss_total, dic_output
