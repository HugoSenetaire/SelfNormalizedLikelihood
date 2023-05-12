import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float


class IsingProposal(nn.Module):
    def __init__(self, dataset):
        super(IsingProposal, self).__init__()
        self.dateset = dataset

    def sample(
        self, nb_sample: int = 1
    ) -> Float[torch.Tensor, "nb_sample nb_point_in_graph"]:
        if nb_sample < len(self.dateset):
            index = np.random.choice(len(self.dateset), nb_sample)

        else:
            index = np.random.choice(len(self.dateset), nb_sample, replace=True)

        center = torch.cat([self.dateset[i][0] for i in index])
        bernoulli_keep = torch.distributions.Bernoulli(
            torch.full_like(center, 0.9)
        ).sample()
        samples = center * bernoulli_keep + (1 - center) * bernoulli_keep

        return samples.detach()
