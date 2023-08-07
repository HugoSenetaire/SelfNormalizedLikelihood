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

    def regul_loss(
        self,
    ):
        if self.cfg.optim_energy.coef_regul < 0.0:
            return 0
        x_gen = self.ebm.proposal.sample(self.num_samples_train).detach()
        energy_samples, dic_output = self.ebm.calculate_energy(x_gen)
        proposal_log_prob = self.ebm.proposal.log_prob(x_gen)
        loss_energy = (energy_samples.flatten() + proposal_log_prob.flatten()).pow(2).mean()
        return loss_energy

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()

        x = batch["data"]
        if self.cfg.train.bias_training_iter > 0:
            self.train_bias(x, ebm_opt, self.cfg.train.bias_training_iter)

        ebm_opt.zero_grad()
        if proposal_opt is not None:
            proposal_opt.zero_grad()
        self.configure_gradient_flow("energy")

        self.stupid_test(batch["data"], suffix="before")

        x = x.requires_grad_()
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        # self.ebm.apply(set_bn_to_eval)
        estimate_log_z, dic, x_gen, x_gen_noisy = self.ebm.estimate_log_z(
            x,
            self.num_samples_train,
            detach_sample=True,
            requires_grad=True,
            return_samples=True,
            noise_annealing=calculate_current_noise_annealing(self.global_step, self.cfg.train.noise_annealing_init, self.cfg.train.noise_annealing_gamma,),
        )
        # self.ebm.apply(set_bn_to_train)

        energy_data, dic_output = self.ebm.calculate_energy(x)

        estimate_log_z = estimate_log_z.mean()
        if (self.cfg.train.start_with_IS_until is not None and self.global_step < self.cfg.train.start_with_IS_until) or ():
            loss_estimate_z = estimate_log_z
        else:
            loss_estimate_z = estimate_log_z.exp() - 1

        loss_energy = energy_data.mean()
        loss_total = loss_energy + loss_estimate_z
        loss_grad_energy = self.gradient_control_l2(x, loss_energy, self.cfg.optim_energy.pg_control_data)
        loss_grad_estimate_z = self.gradient_control_l2(x_gen, loss_estimate_z, self.cfg.optim_energy.pg_control_gen )
        loss_regul_control = self.cfg.optim_energy.coef_regul * self.regul_loss()

        loss_total = (loss_total + loss_grad_energy + loss_grad_estimate_z + loss_regul_control)

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_estimate_z", loss_estimate_z)
        self.log("train/loss_grad_energy", loss_grad_energy)
        self.log("train/loss_grad_estimate_z", loss_grad_estimate_z)
        self.log("train/loss_regul_control", loss_regul_control)
        self.log("train/noise_annealing",calculate_current_noise_annealing(self.global_step,self.cfg.train.noise_annealing_init,self.cfg.train.noise_annealing_gamma,),)

        # Backward ebm
        self.manual_backward(loss_total,)

        # dic_1 = {name : param.grad.norm() for name, param in self.ebm.energy.named_parameters()}
        # for name in dic_1 :
        # print(name, dic_1[name])

        if self.cfg.optim_energy.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.parameters(),
                max_norm=self.cfg.optim_energy.clip_grad_norm,
            )
        # self.log("method/full", 1)
        # self.log("method/no_proposal_zero_grad", 1)
        # self.log("method/nostep", 1)
        proposal_opt.zero_grad()
        ebm_opt.step()
        ebm_opt.zero_grad()
        if proposal_opt is not None:
            proposal_opt.zero_grad()

        self.stupid_test(batch["data"], suffix="after")

        # dic_2 = {name : param.clone().detach().cpu() for name, parawm in self.ebm.base_dist.named_parameters()}

        # for name in dic_1 :
        #     print(name, (dic_1[name] - dic_2[name[]]).abs().max())

        # Update the parameters of the proposal
        self._proposal_step(
            x=x,
            estimate_log_z=None,
            proposal_opt=proposal_opt,
            dic_output=dic_output,
        )

        # Update the parameters of the ebm
        dic_output.update(dic)

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
