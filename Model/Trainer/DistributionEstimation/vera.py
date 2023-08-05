from ...Utils import get_optimizer, get_scheduler
from .abstract_trainer import AbstractDistributionEstimation


class VERA(AbstractDistributionEstimation):
    """
    Trainer using VERA from Grathwohl et al. 2019. reworked from the original implementation of the authors.
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
        self.pg_control = cfg.train.pg_control
        self.entropy_weight = cfg.train.entropy_weight
        assert (
            self.ebm.proposal != self.ebm.base_dist
        ), "The proposal and the base distribution should be different"

    def configure_optimizers(self):
        """
        Change the configure optimizer to not train the sigma in the proposal

        Returns:
        -------
            opt_list (list): The list of optimizers
            sch_list (list): The list of schedulers
        """
        opt_list = super().configure_optimizers()
        parameters_proposal = [
            child.parameters()
            for name, child in self.ebm.proposal.named_children()
            if "signa" not in name
        ]
        proposal_opt = get_optimizer(
            cfg=self.cfg.optim_proposal, list_parameters_gen=parameters_proposal
        )
        # proposal_sch = get_scheduler(cfg=self.cfg, optim=proposal_opt)
        opt_list[1] = proposal_opt
        # if proposal_sch is not None:
        # sch_list = [self.scheduler, proposal_sch]
        return opt_list

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()

        x = batch["data"]
        x_gen, h_gen = self.ebm.proposal.sample_vera(self.num_samples_train)
        x_gen_detach = x_gen.detach()

        # Energy loss :
        energy_data, dic_output = self.ebm.calculate_energy(x)
        energy_gen, dic_output_gen = self.ebm.calculate_energy(x_gen_detach)
        for key in dic_output_gen.keys():
            dic_output[key + "_egen"] = dic_output_gen[key]
        dic_output.update(dic_output_gen)
        loss_energy = energy_data.mean() - energy_gen.mean()

        grad_e_data = (
            torch.autograd.grad(-energy_data.sum(), x, create_graph=True)[0]
            .flatten(start_dim=1)
            .norm(2, 1)
        )

        loss_total = loss_energy + self.pg_control * (grad_e_data**2.0 / 2.0).mean()
        self.log("train_loss", loss_total)

        # Backward ebm
        ebm_opt.zero_grad()
        self.manual_backward(
            loss_total,
            retain_graph=False,
        )
        ebm_opt.step()

        # Update the parameters of the proposal
        energy_gen, _ = self.ebm.calculate_energy(x_gen)
        grad_e_gen = torch.autograd.grad(-energy_gen.sum(), x_gen, retain_graph=True)[0]
        ebm_gn = grad_e_gen.norm(2, 1).mean()
        dic_output["ebm_gn"] = ebm_gn
        if self.entropy_weight != 0.0:
            entropy_obj, ent_gn = self.ebm.proposal.entropy_obj(x_gen, h_gen)
            dic_output["ent_gn"] = ent_gn

        logq_obj = -energy_gen.mean() + self.entropy_weight * entropy_obj

        proposal_loss = logq_obj
        dic_output["proposal_loss"] = proposal_loss

        proposal_opt.zero_grad()
        proposal_loss.backward()
        proposal_opt.step()

        # Update the parameters of the ebm

        self.post_train_step_handler(
            x,
            dic_output,
        )

        return loss_total
