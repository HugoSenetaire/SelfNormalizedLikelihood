from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributions import poisson
from torch.utils.data import Dataset


class Poisson(nn.Module):
    """Module that implements a uniform Poisson distribution. Default parameter is 10. Looks like a Normal"""

    def __init__(
        self, input_size: Tuple[int], dataset: Dataset, lambda_: float = 10.0
    ) -> None:
        super(Poisson, self).__init__()
        self.input_size = input_size
        self.distribution = poisson.Poisson(lambda_)

    def sample(self, nb_sample: int = 1):
        samples = self.distribution.sample((nb_sample, 1, 1))
        return samples

    def log_prob(self, x: Float[torch.Tensor, "batch_size 1 1"]):
        return self.distribution.log_prob(x).flatten(1).sum(1)
