from math import prod
from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float


class EnergyPoissonDistribution(nn.Module):
    """Implement the energy of a poisson distribution. C.f. Oops I took a gradient from Grathwohl et al.

    According to table 1 of the paper the energy is defined as:

    p(x) = exp(-E(x)) / Z

    with E(x) = log(Gamma(x+1)) - x*log(lambda)

    where Gamma is the gamma function.

    Args:
        input_size: tuple of ints, size of the input. This has to be a tuple of size 1.
        lambda_: float, the real valued parameter of the poisson distribution
        learn_lambda: bool, whether to learn lambda or not

    Attributes:
        lambda_: nn.Parameter that represents the real valued parameter of the poisson distribution.
        It is initialized with a random value sampled from U(0,1) if not specified.

    Raises:
        AssertionError: if the input_size is not a tuple of size 1.

    """

    def __init__(
        self,
        input_size: Tuple[int],
        lambda_: float,
        learn_lambda: bool = False,
    ) -> None:
        super(EnergyPoissonDistribution, self).__init__()
        assert prod(input_size) == 1
        if lambda_ is None:
            lambda_ = torch.rand(1) + 1e-3
        lambda_ = torch.Tensor(lambda_)
        self.lambda_ = nn.parameter.Parameter(lambda_, requires_grad=learn_lambda)

    def forward(
        self, x: Float[torch.Tensor, "batch_size 1"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the poisson distribution

        Args:
            x: Float[torch.Tensor, "batch_size 1"], batch input of the energy.

        Returns:
            Float[torch.Tensor, "batch_size 1"], E(x), the energy of the poisson distribution
        """

        energy = torch.lgamma(x + 1) - x * torch.log(self.lambda_)
        return energy.reshape(-1, 1)
