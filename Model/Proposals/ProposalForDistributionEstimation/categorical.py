from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributions import categorical
from torch.utils.data import Dataset


class Categorical(nn.Module):
    """Module that implements a uniform categorical distribution"""

    def __init__(self, input_size: Tuple[int], dataset: Dataset) -> None:
        super(Categorical, self).__init__()
        self.input_size = input_size
        self.num_classes = dataset.num_classes
        self.distribution = categorical.Categorical(
            torch.ones(self.num_classes) / self.num_classes
        )

    def sample(self, nb_sample: int = 1):
        samples = self.distribution.sample((nb_sample, 1, 1))
        return samples

    def log_prob(self, x: Float[torch.Tensor, "batch_size 1 1"]):
        return self.distribution.log_prob(x).flatten(1).sum(1)
