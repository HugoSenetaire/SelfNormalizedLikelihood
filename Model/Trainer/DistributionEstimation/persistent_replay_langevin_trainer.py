import logging
import os
import random

import numpy as np
import torch
import torchvision
import wandb

from ...Sampler.Langevin.langevin import langevin_step
from .abstract_trainer import AbstractDistributionEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to("cpu")
        class_ids = class_ids.detach().to("cpu")

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample.detach(), class_id))
            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device="cuda"):
        samples, class_ids = zip(*random.sample(self.buffer, k=n_samples))
        # np.random.choice
        # items = np.random.choices(self.buffer, k=n_samples)
        # samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids




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
        self.size_replay_buffer = cfg.train.size_replay_buffer
        self.prop_replay_buffer = cfg.train.prop_replay_buffer
        self.nb_steps_langevin = cfg.train.nb_steps_langevin
        self.step_size_langevin = cfg.train.step_size_langevin
        self.sigma_langevin = cfg.train.sigma_langevin
        self.clip_max_norm = cfg.train.clip_max_norm
        self.clip_max_value = cfg.train.clip_max_value
        self.replay_buffer = SampleBuffer(max_samples=self.size_replay_buffer)
        self.save_buffer_every = cfg.train.save_buffer_every
        self.sigma_data = cfg.train.sigma_data

        assert self.ebm.proposal is not None, "The proposal should not be None"

    def get_initial_state(self, x):
        if len(self.replay_buffer) < self.num_samples_train:  # In the original code, it's batch size here
            x_init = self.ebm.proposal.sample(self.num_samples_train)
            id_init = torch.randint(0, 10, (self.num_samples_train,), device=x.device)
        else :
            n_replay = (np.random.rand(self.num_samples_train) < self.prop_replay_buffer).sum()
            replay_sample, replay_id = self.replay_buffer.get(n_replay)
            random_sample = self.ebm.proposal.sample(self.num_samples_train - n_replay)
            random_id = torch.randint(0, 10, (self.num_samples_train - n_replay,), device=x.device)
            x_init = torch.cat([replay_sample, random_sample], 0)
            id_init = torch.cat([replay_id, random_id], 0)

        return x_init, id_init


    def training_energy(self, x):
        f_theta_opt, explicit_bias_opt, base_dist_opt, proposal_opt = self.optimizers
        f_theta_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        if self.sigma_data>0:
            x_data = x + torch.randn_like(x) * self.sigma_data
        else :
            x_data = x

        x_init, id_init = self.get_initial_state(x=x_data)
        for k in range(self.nb_steps_langevin):
            x_init = langevin_step(
                x_init=x_init,
                energy=lambda x: self.ebm.calculate_energy(x, None)[0],
                step_size=self.step_size_langevin,
                sigma=self.sigma_langevin,
                clip_max_norm=self.clip_max_norm,
                clip_max_value=self.clip_max_value,
            ).detach()
            x_init.clamp_(-1, 1)
        self.replay_buffer.push(x_init, id_init)

        energy_data, dic_output = self.ebm.calculate_energy(x_data)
        energy_samples, dic = self.ebm.calculate_energy(x_init)
        for key in dic.keys():
            dic_output[key.replace("data", "samples")] = dic[key]
        loss_total = torch.mean(energy_data) - torch.mean(energy_samples)
        loss_total = self.grads_and_reg(loss_total=loss_total, x=x, x_gen=x_init, energy_data=energy_data, energy_samples=energy_samples,)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", torch.mean(energy_data))
        self.log("train/loss_samples", torch.mean(energy_samples))

        f_theta_opt.step()
        explicit_bias_opt.step()
        if self.train_base_dist:
            base_dist_opt.step()
        f_theta_opt.zero_grad()
        base_dist_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        if self.current_step % self.save_buffer_every == 0:
            images, _ = self.replay_buffer.get(64)
            grid = torchvision.utils.make_grid(images,)
            image = wandb.Image(grid, caption="{}_{}.png".format("buffer", self.current_step))
            self.logger.log({"{}.png".format("buffer",): image},step=self.current_step,)

        return loss_total, dic_output
