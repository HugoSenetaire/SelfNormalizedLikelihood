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
        complete_dataset=None,
    ):
        super().__init__(
            ebm=ebm,
            cfg=cfg,
            complete_dataset=complete_dataset,
        )

    def training_energy(self,x):
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
        estimate_log_z = estimate_log_z.mean()
        if (
            self.cfg.train.start_with_IS_until is not None
            and self.current_step < self.cfg.train.start_with_IS_until
        ) :
            loss_estimate_z = estimate_log_z
        else:
            loss_estimate_z = estimate_log_z.exp() - 1

 
        loss_energy = energy_data.mean()
        loss_total = loss_energy + loss_estimate_z

        aux_data = x.detach()
        aux_data.requires_grad_(True)
        f_theta_data = self.ebm.energy(aux_data).mean()
        f_theta_data.backward(retain_graph=True)
        loss_grad_energy = self.gradient_control_l2(
            aux_data, -f_theta_data, self.cfg.optim_energy.pg_control_data
        )

        aux_gen = x_gen.detach()
        aux_gen.requires_grad_(True)
        f_theta_gen = self.ebm.energy(aux_gen).mean()
        f_theta_gen.backward(retain_graph=True)
        loss_grad_estimate_z = self.gradient_control_l2(
            aux_gen, -f_theta_gen, self.cfg.optim_energy.pg_control_gen
        )

        min_data_len = min(x.shape[0], x_gen.shape[0])
        epsilon = torch.rand(min_data_len, device=x.device)
        for i in range(len(x.shape) - 1):
            epsilon = epsilon.unsqueeze(-1)
        epsilon = epsilon.expand(min_data_len, *x.shape[1:])
        aux_2 = (epsilon.sqrt() * x[:min_data_len,] + (1 - epsilon).sqrt() * x_gen[:min_data_len]).detach()
        aux_2.requires_grad_(True)
        f_theta_gen_2 = self.ebm.energy(aux_2).mean()
        f_theta_gen_2.backward(retain_graph=True)
        loss_grad_estimate_mix = self.gradient_control_l2(
            aux_2, -f_theta_gen_2, self.cfg.optim_energy.pg_control_mix
        )


        loss_total = (loss_total + loss_grad_energy + loss_grad_estimate_z+loss_grad_estimate_mix)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_estimate_z", loss_estimate_z)
        self.log("train/loss_grad_energy", loss_grad_energy)
        self.log("train/loss_grad_estimate_z", loss_grad_estimate_z)
        self.log("train/loss_grad_estimate_mix", loss_grad_estimate_mix)
        self.log("train/noise_annealing",current_noise_annealing,)

        # Backward ebm
        self.manual_backward(loss_total,retain_graph=True,)

        # for param in self.ebm.explicit_bias :
            # print(param.grad)
       
        if self.cfg.optim_energy.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.parameters(),
                max_norm=self.cfg.optim_energy.clip_grad_norm,
            )

        energy_opt.step()
        if self.train_base_dist :
            base_dist_opt.step()
        energy_opt.zero_grad()
        base_dist_opt.zero_grad()
        proposal_opt.zero_grad()

        # if (
        #     self.cfg.train.start_with_IS_until is not None
        #     and self.current_step < self.cfg.train.start_with_IS_until
        # ) :
        #     for param in self.ebm.energy.parameters():
        #         param.requires_grad = False
        #     for param in self.ebm.explicit_bias.parameters():
        #         param.requires_grad = True
            
            
        #     estimate_log_z, dic, x_gen, x_gen_noisy = self.ebm.estimate_log_z(
        #         x,
        #         self.num_samples_train,
        #         detach_sample=True,
        #         requires_grad=False,
        #         return_samples=True,
        #         noise_annealing=current_noise_annealing,
        #     )
        #     loss_estimate_z = (estimate_log_z.exp()-1).mean()
        #     self.manual_backward(loss_estimate_z,retain_graph=False,)
        #     energy_opt.step()

            
        return loss_total, dic_output
