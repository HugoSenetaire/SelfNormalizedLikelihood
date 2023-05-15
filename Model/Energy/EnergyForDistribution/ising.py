from math import prod
from typing import Tuple

import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch.distributions import bernoulli


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
        n_node: int,
        average_degree: int = 4,
        init_bias: float = 0.0,
        learn_G: bool = False,
        learn_bias: bool = False,
    ) -> None:
        super(ErdosRenyiEnergyIsing, self).__init__()
        # Code from Oops I took a gradient
        # g = ig.Graph.Erdos_Renyi(n_node, float(average_degree) / float(n_node))
        # A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        # A = torch.tensor(A).float()
        A = torch.randn((n_node, n_node)) * 0.01
        weights = torch.randn_like(A) * ((1.0 / average_degree) ** 0.5)
        weights = weights * (1 - torch.tril(torch.ones_like(weights)))
        weights = weights + weights.t()

        self.G = nn.Parameter(A * weights, requires_grad=learn_G)
        self.bias = nn.Parameter(
            torch.ones((n_node,)).float() * init_bias, requires_grad=learn_bias
        )
        self.data_dim = n_node

    @property
    def J(self):
        return self.G

    def forward(
        self, x: Float[torch.Tensor, "batch_size nb_point_in_graph"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the Ising model.

        Args:
            x: Float[torch.Tensor, "batch_size nb_point_in_graph"], batch input of the energy.
            x is a vector of zeros and ones.

        Returns:
            Float[torch.Tensor, "batch_size"], E(x), the energy of the Ising model.
        """
        # code from Oops I took a gradient

        x = 2 * x - 1  # convert 0/1 to -1/1
        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return -xgx - b
