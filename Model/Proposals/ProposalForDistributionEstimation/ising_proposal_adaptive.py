import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn.parameter import Parameter
from .ising_proposal import IsingProposal

class IsingProposalAdaptive(IsingProposal):
    """Proposal for Ising model

    Attributes:
        dataset: torch.utils.data.Dataset, dataset of the Ising model
        p: float, probability of not flipping the spin
    """

    def __init__(self, default_proposal, input_size, dataset, p: float = 0.9):
        super(IsingProposalAdaptive, self).__init__(input_size =input_size, dataset=dataset, centers=None, p = p)
        self.x = None
    def set_x(self, x):
        self.x = x

    def get_center(self, index):
        # Might be worth it time wise to store the samples in memory rather than recalculating them everytime
        if self.center is not None :
            return self.center[index]
        else :
            return torch.stack([self.dataset[i][0] for i in index])

    def sample(self, nb_sample: int = 1) -> Float[torch.Tensor, "nb_sample nb_nodes"]:
        if self.x is not None :
            aux_ising = IsingProposal(self.x.shape[1], self.dataset, p = self.p, centers=self.x)
            return aux_ising.sample(nb_sample).detach()
        else :
            return super().sample(nb_sample).detach()

    def log_prob(
        self, x: Float[torch.Tensor, "batch_size nb_nodes"]
    ) -> Float[torch.Tensor, "batch_size"]:
        if self.x is not None :
            aux_ising = IsingProposal(self.x.shape[1], self.dataset, p = self.p, centers=self.x)
            return aux_ising.log_prob(x)
        else :
            return super().log_prob(x)
