import torch
import logging

from .abstract_trainer import AbstractDistributionEstimation
from ...Utils.noise_annealing import calculate_current_noise_annealing


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

        self.configure_gradient_flow("energy")
        if self.train_base_dist :
            for param in self.ebm.base_dist.parameters():
                param.requires_grad = True
        current_noise_annealing = calculate_current_noise_annealing(
                self.current_step,
                self.cfg.train.noise_annealing_init,
                self.cfg.train.noise_annealing_gamma,
            )

        x = x.requires_grad_()

        batch_size = x.shape[0]
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

        samples = self.ebm.sample(self.num_samples_train).to(x.device, x.dtype)
        energy_samples = self.ebm.energy(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
        if self.ebm.explicit_bias :
            energy_samples = self.ebm.explicit_bias(energy_samples)
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
        logp_gen = -energy_samples.reshape(self.num_samples_train,1) # logp(x̃)
        logq_gen = log_prob_proposal_samples.reshape(self.num_samples_train,1)  # logq(x̃)
        log_noise_ratio = torch.log(torch.tensor(self.num_samples_train/batch_size, dtype=energy_data.dtype, device=energy_data.device))

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
        nce_objective = value_data.mean() + (value_gen + log_noise_ratio).mean() ## SHOULD I DO BOTH TIMES THE ADDING OF LOG NOISE RATIO
        # nce_objective = value_data.mean() + value_gen.mean() ## SHOULD I DO BOTH TIMES THE ADDING OF LOG NOISE RATIO
        nce_loss = -nce_objective
        
        
        min_data_len = min(x.shape[0], samples.shape[0])
        epsilon = torch.rand(min_data_len, device=x.device)
        for i in range(len(x.shape) - 1):
            epsilon = epsilon.unsqueeze(-1)
        epsilon = epsilon.expand(min_data_len, *x.shape[1:])
        aux_2 = (epsilon.sqrt() * x[:min_data_len,] + (1 - epsilon).sqrt() * samples[:min_data_len]).detach()
        aux_2.requires_grad_(True)
        f_theta_gen_2 = self.ebm.energy(aux_2).mean()
        f_theta_gen_2.backward(retain_graph=True)
        loss_grad_estimate_mix = self.gradient_control_l2(
            aux_2, -f_theta_gen_2, self.cfg.optim_energy.pg_control_mix
        )

        loss_total = nce_loss + loss_grad_estimate_mix
        # loss_total = nce_loss

        self.log("train/loss_grad_estimate_mix", loss_grad_estimate_mix)
        self.log("train/loss_nce", nce_loss)
        self.log("train/loss_total", loss_total)

        self.manual_backward(loss_total)
        energy_opt.step()
        if self.train_base_dist:
            base_dist_opt.step()

        return loss_total, dic_output

