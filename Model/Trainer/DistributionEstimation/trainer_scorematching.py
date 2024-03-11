import logging

import torch

from .abstract_trainer import AbstractDistributionEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class ScoreMatchingTrainer(AbstractDistributionEstimation):
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

    def exact_score_matching(
        self,
        data,
    ):
        data = data.flatten(1)
        data.requires_grad_(True)
        energy_data, dic = self.ebm.calculate_energy(data)
        logp = -energy_data.reshape(data.shape[0]).sum(-1)  # logp(x)
        grad1 = torch.autograd.grad(logp, data, create_graph=True)[0]
        loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.0
        loss2 = torch.zeros(data.shape[0], device=data.device)

        iterator = range(data.shape[1])

        for i in iterator:
            if self.training:
                grad = torch.autograd.grad(
                    grad1[:, i].sum(), data, create_graph=True, retain_graph=True
                )[0][:, i]
            if not self.training:
                grad = torch.autograd.grad(
                    grad1[:, i].sum(), data, create_graph=False, retain_graph=True
                )[0][:, i]
                grad = grad.detach()
            loss2 += grad

        loss = loss1 + loss2

        return loss, dic

    def training_step(self, batch, batch_idx):
        energy_opt, base_dist_opt, proposal_opt = self.optimizers


        x = batch['data']
        if hasattr(self.proposal, 'set_x'):
            self.proposal.set_x(x)

        loss_total, dic_output = self.exact_score_matching(x)
        self.log("train_loss", loss_total.mean())
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total.mean(),
            retain_graph=False,
        )
        ebm_opt.step()
        ebm_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.num_samples_train)
        estimate_log_z = estimate_log_z.mean()
        dic_output.update(dic)

        # Update the parameters of the proposal
        self._proposal_step(x = x, estimate_log_z = estimate_log_z, proposal_opt = proposal_opt, dic_output=dic_output,)

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
