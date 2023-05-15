import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn.parameter import Parameter


class IsingProposal(nn.Module):
    """Proposal for Ising model

    Attributes:
        dataset: torch.utils.data.Dataset, dataset of the Ising model
        p: float, probability of not flipping the spin
    """

    def __init__(self, input_size, dataset, centers = None, p: float = 0.9):
        super(IsingProposal, self).__init__()
        self.dataset = dataset
        self.p = p
        self.dummy_param = Parameter(torch.Tensor([0.0]), requires_grad=False)

        shape = self.dataset[0][0].shape
        len_dataset = len(self.dataset)
        if centers is not None:
            # For the adaptive proposal, we can pass the centers directly
            self.centers = centers
            self.len_centers = len(centers)
        else :
            self.len_centers = len_dataset
            if np.prod(shape)*len(dataset)<1e5:
                # Checking the full size of storing everything :
                self.centers = torch.stack([self.dataset[i][0] for i in range(len(self.dataset))])
            else :
                self.centers = None

    def get_centers(self, index):
        # Might be worth it time wise to store the samples in memory rather than recalculating them everytime
        if self.centers is not None :
            return self.centers[index]
        else :
            return torch.stack([self.dataset[i][0] for i in index])

    def sample(self, nb_sample: int = 1) -> Float[torch.Tensor, "nb_sample nb_nodes"]:
        if nb_sample < self.len_centers:
            index = np.random.choice(self.len_centers, nb_sample)
        else:
            index = np.random.choice(self.len_centers, nb_sample, replace=True)

        center = self.get_centers(index)
        bernoulli_keep = torch.distributions.Bernoulli(
            torch.full_like(center, self.p)
        ).sample()

        samples = center * bernoulli_keep + (1 - center) * (1 - bernoulli_keep)
        return samples.detach()

    def log_prob(
        self, x: Float[torch.Tensor, "batch_size nb_nodes"]
    ) -> Float[torch.Tensor, "batch_size"]:

        data = self.get_centers(range(self.len_centers)).unsqueeze(1)
        x_expanded = x.unsqueeze(0)  # 1, batch_size, nb_nodes

        dependency = (data - x_expanded).abs()  # len(dataset), batch_size, nb_nodes
        log_prob = (
            torch.distributions.Bernoulli(torch.full_like(dependency, self.p))
            .log_prob(dependency)
            .sum(-1)
        )  # len(dataset), batch_size
        log_prob = log_prob.logsumexp(0) - torch.log(
            torch.tensor(self.len_centers)
        )  # batch_size
        return log_prob
