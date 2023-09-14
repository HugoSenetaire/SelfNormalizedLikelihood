import torch

from ...Utils.noise_annealing import calculate_current_noise_annealing
from .abstract_trainer import AbstractDistributionEstimation


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def set_bn_to_train(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.train()


class KALE(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal.
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

    def training_energy(self, x):
        # Get parameters
        energy_opt, base_dist_opt, proposal_opt = self.optimizers_perso()
        energy_opt.zero_grad()
        base_dist_opt.zero_grad()
        proposal_opt.zero_grad()
        dic_output = {}

        self.configure_gradient_flow("energy")

        energy_data = self.ebm.f_theta(x)
        bias = self.ebm.explicit_bias(x)

        sample = self.ebm.proposal.sample(self.num_samples_train)
        energy_sample = self.ebm.f_theta(sample)

        loss_data = -energy_data.mean() - bias

        loss_sample = (-energy_sample + bias).logsumexp(0) - torch.log(
            torch.tensor(self.num_samples_train)
        )
        loss_sample = loss_sample.exp()

        loss_total = loss_data + loss_sample

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_data)
        self.log("train/loss_estimate_z", loss_sample)

        # self.log("train/loss_grad_energy", loss_grad_energy)
        # self.log("train/loss_grad_estimate_z", loss_grad_estimate_z)
        # self.log("train/noise_annealing",current_noise_annealing,)

        # Backward ebm
        self.manual_backward(
            loss_total,
            retain_graph=True,
        )

        if self.cfg.optim_energy.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.parameters(),
                max_norm=self.cfg.optim_energy.clip_grad_norm,
            )

        energy_opt.step()
        return loss_total, dic_output
