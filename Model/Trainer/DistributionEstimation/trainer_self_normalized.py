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

        x = batch["data"]
        if self.pg_control > 0:
            x = x.requires_grad_()
        if hasattr(self.ebm.proposal, "set_x"):
            self.ebm.proposal.set_x(x)

        # self.ebm.apply(set_bn_to_eval)
        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.num_samples_train)
        # self.ebm.apply(set_bn_to_train)

        energy_samples, dic_output = self.ebm.calculate_energy(x)

        estimate_log_z = estimate_log_z.mean()
        loss_estimate_z = estimate_log_z.exp()

        loss_energy = energy_samples.mean()
        loss_total = loss_energy + loss_estimate_z

        loss_grad_energy = self.gradient_control_l2(x, loss_energy)
        # loss_grad_estimate_z = self.gradient_control_l2(x, loss_estimate_z)
        loss_total = loss_total + loss_grad_energy

        self.log("train_loss", loss_total)
        self.log("train_loss_energy", loss_energy)
        self.log("train_loss_estimate_z", loss_estimate_z)
        self.log("train_loss_grad_energy", loss_grad_energy)
        # self.log("train_loss_grad_estimate_z", loss_grad_estimate_z)

        # Backward ebmxx
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total,
            retain_graph=True,
        )

        # torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)

        # Update the parameters of the proposal
        self._proposal_step(
            x=x,
            estimate_log_z=estimate_log_z,
            proposal_opt=proposal_opt,
            dic_output=dic_output,
        )

        # Update the parameters of the ebm
        ebm_opt.step()
        dic_output.update(dic)

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
