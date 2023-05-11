from math import prod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class ErdosRenyiEnergyIsing(nn.Module):
    """Implement the energy of an Ising model. C.f. Oops I took a gradient from Grathwohl et al.

    According to table 1 of the paper the energy is defined as:

    p(x) = exp(-E(x)) / Z

    with E(x) = - x^T@W@x - b^T@x


    Args:
        input_size: Tuple[int], size of the input. This will be flatten to a tuple of size 1.
        learn_W: bool, whether to learn W or not
        learn_b: bool, whether to learn b or not


    Attributes:
        W: nn.Linear (input_size, hidden_dim), the parameters of the energy.
            W is initialized as the adjacency matrix of an Erdos-Renyi graph with probability p=4/prod(input_size).
            So each node has an average degree of 4.
        b: torch.Tensor of size (hidden_dim), the parameters of the energy.
            b is initialized as a tensor of ones.

    """

    def __init__(
        self,
        input_size: Tuple[int],
        learn_W: bool = True,
        learn_b: bool = True,
    ) -> None:
        super().__init__()
        N = prod(input_size)
        assert N > 4, "input_size is too small"
        p = 1 / (N - 1)
        G = torch.rand(N, N) < p
        G = torch.triu(G, diagonal=1)
        G = (G + G.T) * 1.0

        weights = torch.randn_like(G) * (
            (1.0 / (N * p)) ** 0.5
        )  # From Oops I took a gradient
        weights = weights * (1 - torch.tril(torch.ones_like(weights)))
        weights = weights + weights.t()

        self.W = nn.parameter.Parameter(G * weights, requires_grad=learn_W)
        self.b = nn.parameter.Parameter(torch.ones(N), requires_grad=learn_b)

    def forward(
        self, x: Float[torch.Tensor, "batch_size *dim"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the Ising model.

        Args:
            x: Float[torch.Tensor, "batch_size *dim"], batch input of the energy.

        Returns:
            Float[torch.Tensor, "batch_size"], E(x), the energy of the Ising model.
        """
        x = x.flatten(1)
        Wx = torch.matmul(x, self.W.T)
        return -torch.sum(x * Wx, dim=1) - x @ self.b
