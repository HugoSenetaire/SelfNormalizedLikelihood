import torch

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

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()
        ebm_opt.zero_grad()
        if proposal_opt is not None:
            proposal_opt.zero_grad()
        self.configure_gradient_flow('energy')

        x = batch["data"]
        x = x.requires_grad_()
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        # self.ebm.apply(set_bn_to_eval)
        estimate_log_z, dic, x_gen = self.ebm.estimate_log_z(
            x,
            self.num_samples_train,
            detach_sample=True,
            requires_grad=True,
            return_samples=True,
        )
        # self.ebm.apply(set_bn_to_train)

        energy_samples, dic_output = self.ebm.calculate_energy(x)

        estimate_log_z = estimate_log_z.mean()
        if (self.cfg.train.start_with_IS_until is not None and self.global_step < self.cfg.train.start_with_IS_until):
            loss_estimate_z = estimate_log_z
        else:
            loss_estimate_z = estimate_log_z.exp()

        loss_energy = energy_samples.mean()
        loss_total = loss_energy + loss_estimate_z

        loss_grad_energy = self.gradient_control_l2(
            x, loss_energy, self.cfg.optim_energy.pg_control_data
        )
        loss_grad_estimate_z = self.gradient_control_l2(
            x_gen, loss_estimate_z, self.cfg.optim_energy.pg_control_gen
        )
        loss_total = loss_total + loss_grad_energy + loss_grad_estimate_z

        self.log("train/loss_total", loss_total)
        self.log("train/loss_energy", loss_energy)
        self.log("train/loss_estimate_z", loss_estimate_z)
        self.log("train/loss_grad_energy", loss_grad_energy)
        self.log("train/loss_grad_estimate_z", loss_grad_estimate_z)

        # Backward ebm
        self.manual_backward(
            loss_total,
        )
        if self.cfg.optim_energy.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.ebm.parameters(),
                max_norm=self.cfg.optim_energy.clip_grad_norm,
            )
        proposal_opt.zero_grad()
        ebm_opt.step()
        ebm_opt.zero_grad()
        if proposal_opt is not None:
            proposal_opt.zero_grad()

        torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)

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
