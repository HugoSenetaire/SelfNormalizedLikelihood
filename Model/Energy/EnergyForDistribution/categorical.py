from math import prod
from typing import List, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float


class EnergyCategoricalDistrib(nn.Module):
    """Implement the energy of a categorical distribution. C.f. Oops I took a gradient from Grathwohl et al.

    According to table 1 of the paper the energy is defined as:

    p(x) = exp(-E(x)) / Z

    with E(x) = - x^T@theta

    Args:
        input_size: tuple of ints, size of the input. This will be flatten to a tuple of size one.
        theta: List[float], the parameters of the categorical distribution. Default value is uniform.
        learn_theta: bool, whether to learn theta or not

    Attributes:
        theta: Float[torch.Tensor, "dim"], the parameters of the categorical distribution

    Raises:
        AssertionError: if input_size and theta do not match
        AssertionError: if theta does not sum to one.

    """

    def __init__(
        self,
        input_size: Tuple[int],
        theta: List[float] = None,
        learn_theta: bool = False,
    ) -> None:
        super().__init__()
        assert prod(input_size) == len(theta), "input_size and theta do not match"
        if theta is None:
            theta = [1 / prod(input_size)] * prod(input_size)
        theta_ = torch.Tensor(theta)
        self.theta = nn.parameter.Parameter(theta_, requires_grad=learn_theta)

    def forward(
        self, x: Float[torch.Tensor, "batch_size *dim"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the categorical distribution.

        Args:
            x: Float[torch.Tensor, "batch_size *dim"], the input of the energy. This is a batch of vectors of size dim.

        Returns:
            Float[torch.Tensor, "batch_size"], E(x), the energy of the categorical distribution.
        """
        x = x.flatten(1)
        return x @ self.theta
