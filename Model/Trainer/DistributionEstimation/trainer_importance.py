from .abstract_trainer import AbstractDistributionEstimation


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
        nb_sample_train_estimate=1024,
        **kwargs,
    ):
        super().__init__(
            ebm=ebm,
            cfg=cfg,
            complete_data=complete_dataset,
            nb_sample_train_estimate=nb_sample_train_estimate,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()
        x = batch['data']
        if hasattr(self.ebm.proposal, 'set_x'):
            self.ebm.proposal.set_x(x)
        energy_samples, dic_output = self.ebm.calculate_energy(x)

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.num_samples_train)
        estimate_log_z = estimate_log_z.mean()
        loss_estimate_z = estimate_log_z

        loss_energy = energy_samples.mean()
        loss_total = loss_energy + loss_estimate_z
        self.log("train_loss", loss_total)

        # Backward ebmxx
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total,
            retain_graph=True,
        )

        # Update the parameters of the proposal
        self._proposal_step(x = x, estimate_log_z = estimate_log_z, proposal_opt = proposal_opt, dic_output=dic_output,)

        # Update the parameters of the ebm
        ebm_opt.step()
        dic_output.update(dic)

        self.post_train_step_handler(x,dic_output,)

        return loss_total
