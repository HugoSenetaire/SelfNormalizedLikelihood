from math import prod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class EnergyRBM(nn.Module):
    """Implement the energy of a Random Boltzmann Machine. C.f. Oops I took a gradient from Grathwohl et al.

    According to table 1 of the paper the energy is defined as:

    p(x) = exp(-E(x)) / Z

    with E(x) = - Sum_i Softplus(Wx +b)_i - c^t@x


    Args:
        input_size: Tuple[int], size of the input. This will be flatten to a tuple of size 1.
        hidden_dim: int, number of hidden units

    Attributes:
        W: torch.Tensor of size (input_size, hidden_dim), the parameters of the energy.
            W is sampled from N(0, .05I)
        b: torch.Tensor of size (hidden_dim), the parameters of the energy.
            b is sampled from N(0,I)
        c: torch.Tensor of size (input_size), the parameters of the energy.
            c is sampled from N(0,I)

    Note:
        In oops i took a gradient, hidden_dim is 500 for MNIST, 100 otherwise.
        Samples are initialized from uniform Bernoulli distribution. Samplers are run for 2000 steps.

    """

    def __init__(
        self,
        input_size: Tuple[int],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        assert len(input_size) == 1
        self.W = nn.parameter.Parameter(
            torch.randn(hidden_dim, prod(input_size)) * torch.sqrt(0.05)
        )
        self.b = nn.parameter.Parameter(torch.randn(hidden_dim))
        self.c = nn.parameter.Parameter(torch.randn(prod(input_size)))

    def forward(
        self, x: Float[torch.Tensor, "batch_size *dim"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the RBM

        Args:
            x: Float[torch.Tensor, "batch_size *dim"], batch input of the energy.

        Returns:
            Float[torch.Tensor, "batch_size"], E(x), the energy of the RBM
        """
        x = x.flatten(1)
        return (
            -torch.sum(F.softplus(torch.matmul(x, self.W.T) + self.b), dim=1)
            - x @ self.c
        )
