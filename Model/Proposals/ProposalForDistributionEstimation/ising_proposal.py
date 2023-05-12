import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float


class IsingProposal(nn.Module):
    """Proposal for Ising model

    Attributes:
        dataset: torch.utils.data.Dataset, dataset of the Ising model
        p: float, probability of not flipping the spin
    """

    def __init__(self, dataset, p: float = 0.9):
        super(IsingProposal, self).__init__()
        self.dateset = dataset
        self.p = p

    def sample(self, nb_sample: int = 1) -> Float[torch.Tensor, "nb_sample nb_nodes"]:
        if nb_sample < len(self.dateset):
            index = np.random.choice(len(self.dateset), nb_sample)
        else:
            index = np.random.choice(len(self.dateset), nb_sample, replace=True)

        center = torch.cat([self.dateset[i][0] for i in index])
        bernoulli_keep = torch.distributions.Bernoulli(
            torch.full_like(center, self.p)
        ).sample()
        samples = center * bernoulli_keep + (1 - center) * (1 - bernoulli_keep)

        return samples.detach()

    def log_prob(
        self, x: Float[torch.Tensor, "batch_size nb_nodes"]
    ) -> Float[torch.Tensor, "batch_size"]:
        data = torch.cat(
            [self.dataset[i][0] for i in range(len(self.dataset))]
        ).unsqueeze(
            1
        )  # len(dataset), 1, nb_nodes
        x_expanded = x.unsqueeze(0)  # 1, batch_size, nb_nodes

        dependency = (data - x_expanded).abs()  # len(dataset), batch_size, nb_nodes
        log_prob = (
            torch.distributions.Bernoulli(torch.full_like(dependency, self.p))
            .log_prob(dependency)
            .sum(-1)
        )  # len(dataset), batch_size
        log_prob = log_prob.logsumexp(0) - torch.log(
            torch.tensor(len(self.dateset))
        )  # batch_size
        return log_prob
