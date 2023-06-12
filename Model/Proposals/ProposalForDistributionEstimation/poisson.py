from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributions import poisson
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from .abstract_proposal import AbstractProposal


class Poisson(AbstractProposal):
    """Module that implements a uniform Poisson distribution. Default parameter is 10. Looks like a Normal"""

    def __init__(
        self, input_size: Tuple[int], dataset: Dataset, lambda_: float = 10.0
    ) -> None:
        super(Poisson, self).__init__(input_size=input_size)
        self.lambda_ = Parameter(torch.Tensor([lambda_]))

    def sample_simple(self, nb_sample: int = 1):
        samples = (
            poisson.Poisson(self.lambda_).sample((nb_sample,)).type("torch.FloatTensor")
        )
        return samples

    def log_prob_simple(self, x: Float[torch.Tensor, "batch_size 1"]):
        return poisson.Poisson(self.lambda_).log_prob(x).sum(1)
