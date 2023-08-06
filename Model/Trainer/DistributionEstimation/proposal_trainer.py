import logging

import torch

from .abstract_trainer import AbstractDistributionEstimation

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
        assert self.ebm.proposal is not None, "The proposal should not be None"

    def training_step(self, batch, batch_idx):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers_perso()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dic_output = {}
        x = batch["data"].to(device)

        proposal_loss, dic = self._proposal_step(
            x=x,
            estimate_log_z=None,
            proposal_opt=proposal_opt,
            dic_output=dic_output,
        )
        self.log('train/loss_total', proposal_loss)


        self.post_train_step_handler(
            x,
            dic_output,
        )


        return dic_output
