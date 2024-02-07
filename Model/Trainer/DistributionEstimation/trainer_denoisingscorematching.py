import torch
from .abstract_trainer import AbstractDistributionEstimation
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class DenoisingScoreMatchingTrainer(AbstractDistributionEstimation):
    """
    Trainer for an EBM using denoising score matching.
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

        self.sigma = cfg.trainer.sigma 
        logger.warning("DSM not taken in charge yet")
        raise NotImplementedError


    def denoising_score_matching(
        self,
        data,
    ):
        data = data.flatten(1)
        data.requires_grad_(True)
        vector = torch.randn_like(data) * self.sigma
        perturbed_data = data + vector
        energy_perturbed_data, dic = self.ebm.calculate_energy(perturbed_data)
        logp = -energy_perturbed_data  # logp(x)

        dlogp = (
            self.sigma**2
            * torch.autograd.grad(logp.sum(), perturbed_data, create_graph=True)[0]
        )
        kernel = vector
        loss = torch.norm(dlogp + kernel, dim=-1) ** 2
        loss = loss.mean() / 2.0

        return loss, dic

    def training_step(self, x,):
        energy_opt, base_dist_opt, proposal_opt = self.optimizers

        if hasattr(self.ebm.proposal, 'set_x'):
            self.ebm.proposal.set_x(x)

        loss_total, dic_output = self.denoising_score_matching(x)
        self.log("train_loss", loss_total.mean())
        energy_opt.zero_grad()
       
        loss_total.mean().backward()
        energy_opt.step()

        energy_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.num_samples_train)
        estimate_log_z = (estimate_log_z-np.log(self.num_samples_train)).logsumexp(dim=0)
        dic_output.update(dic)

        
        return loss_total, dic_output
