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


class SelfNormalizedTrainer(AbstractDistributionEstimation):
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
        self.fix_proposal()
        self.free_f_theta()
        self.free_log_bias()
        if self.train_base_dist:
            self.free_base_dist()
        else:
            self.fix_base_dist()

            

        # Get parameters
        f_theta_opt, log_bias_opt, base_dist_opt, proposal_opt = self.optimizers
        f_theta_opt.zero_grad()
        log_bias_opt.zero_grad()



        current_noise_annealing = calculate_current_noise_annealing(
            self.current_step,
            self.cfg.train.noise_annealing_init,
            self.cfg.train.noise_annealing_gamma,
        )

        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        estimate_log_z, dic, x_gen, x_gen_noisy = self.ebm.estimate_log_z(
            x,
            self.num_samples_train,
            detach_sample=True,
            requires_grad=False,
            return_samples=True,
            noise_annealing=current_noise_annealing,
        )

        energy_data, dic_output = self.ebm.calculate_energy(x)
        dic_output.update(dic)
        # estimate_log_z = estimate_log_z.mean()
        if (self.cfg.train.start_with_IS_until is not None and self.current_step < self.cfg.train.start_with_IS_until):
            loss_estimate_z = estimate_log_z
        else:
            loss_estimate_z = estimate_log_z.exp() - 1

        loss_energy = energy_data.mean()
        loss_total = loss_energy + loss_estimate_z

        if self.cfg.optim_f_theta.pg_control_mix is not None and self.cfg.optim_f_theta.pg_control_mix > 0:
            min_data_len = min(x.shape[0], x_gen.shape[0])
            epsilon = torch.rand(min_data_len, device=x.device)
            for i in range(len(x.shape) - 1):
                epsilon = epsilon.unsqueeze(-1)
            epsilon = epsilon.expand(min_data_len, *x.shape[1:])
            aux_2 = (epsilon.sqrt() * x[:min_data_len,] + (1 - epsilon).sqrt() * x_gen[:min_data_len]).detach()
            aux_2.requires_grad_(True)
            f_theta_gen_2 = self.ebm.f_theta(aux_2).mean()
            f_theta_gen_2.backward(retain_graph=True)
            loss_grad_estimate_mix = self.gradient_control_l2(aux_2, -f_theta_gen_2, self.cfg.optim_f_theta.pg_control_mix)
            loss_total = loss_total + loss_grad_estimate_mix
            self.log("train/loss_grad_estimate_mix", loss_grad_estimate_mix)


        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_estimate_z", loss_estimate_z)
        self.log("train/noise_annealing",current_noise_annealing,)

        # Backward ebm
        loss_total.backward()


        if self.cfg.optim_f_theta.clip_grad_norm is not None and self.cfg.optim_f_theta.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.parameters(),
                max_norm=self.cfg.optim_f_theta.clip_grad_norm,
            )

        f_theta_opt.step()
        log_bias_opt.step()
        if self.train_base_dist:
            base_dist_opt.step()
        f_theta_opt.zero_grad()
        base_dist_opt.zero_grad()
        log_bias_opt.zero_grad()


        return loss_total, dic_output
