from .abstract_trainer import AbstractDistributionEstimation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class ProposalTrainer(AbstractDistributionEstimation):
    """
    Trainer for the an importance sampling estimator of the partition function, which can be either importance sampling (with log) or self.normalized (with exp).
    Here, the proposal is trained by maximizing the likelihood of the data under the proposal. This is not controlled by the proposal loss of the abstract trainer
    simply because the ebm is not properly defined in this case.
    """

    def __init__(
        self, ebm, cfg, complete_dataset=None,
    ):
        super().__init__(
            ebm=ebm,
            cfg=cfg,
            complete_dataset=complete_dataset,
        )
        assert self.ebm.proposal is not None, "The proposal should not be None"

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()
        x = batch["data"]
        log_prob_proposal_data = self.ebm.proposal.log_prob(
            x,
        )
        self.log("train_proposal_log_likelihood", log_prob_proposal_data.mean())
        loss_proposal = -log_prob_proposal_data.mean()
        self.manual_backward(
            (loss_proposal), inputs=list(self.ebm.proposal.parameters())
        )
        self.log("train_loss", loss_proposal)
        proposal_opt.step()
        # Update the parameters of the ebm
        ebm_opt.step()

        return loss_proposal.mean()
