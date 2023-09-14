import logging

import torch

from ...Sampler.Langevin.langevin import langevin_step
from .abstract_trainer import AbstractDistributionEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class ShortTermLangevin(AbstractDistributionEstimation):
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
        self.nb_steps_langevin = cfg.train.nb_steps_langevin
        self.step_size_langevin = cfg.train.step_size_langevin
        self.sigma_langevin = cfg.train.sigma_langevin
        self.clip_max_norm = cfg.train.clip_max_norm
        self.clip_max_value = cfg.train.clip_max_value
        self.sigma_data = cfg.train.sigma_data

        assert self.ebm.proposal is not None, "The proposal should not be None"

    def training_energy(self, x):
        # assert False
        # Get parameters
        f_theta_opt, explicit_bias_opt, base_dist_opt, proposal_opt = self.optimizers
        f_theta_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        x_data = x + torch.randn_like(x) * self.sigma_data
        x_init = self.ebm.proposal.sample(self.num_samples_train)

        for k in range(self.nb_steps_langevin):
            x_init = langevin_step(
                x_init=x_init,
                energy=lambda x: self.ebm.calculate_energy(x, None)[0],
                step_size=self.step_size_langevin,
                sigma=self.sigma_langevin,
                clip_max_norm=self.clip_max_norm,
                clip_max_value=self.clip_max_value,
            ).detach()

        energy_data, dic_output = self.ebm.calculate_energy(x_data)
        energy_samples, dic = self.ebm.calculate_energy(x_init)

        for key in dic.keys():
            dic_output["samples_" + key] = dic[key]
        loss_energy = torch.mean(energy_data)
        loss_samples = torch.mean(energy_samples)
        loss_total = loss_energy - loss_samples

        self.grads_and_reg(loss_total=loss_total, x=x, x_gen=None)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_samples", loss_samples)

        f_theta_opt.step()
        explicit_bias_opt.step()
        if self.train_base_dist:
            base_dist_opt.step()
        f_theta_opt.zero_grad()
        base_dist_opt.zero_grad()
        explicit_bias_opt.zero_grad()

        return loss_total, dic_output
