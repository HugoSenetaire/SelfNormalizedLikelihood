from math import prod
from typing import List, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float


def get_EnergyCategoricalDistrib(input_size, cfg):
    return EnergyCategoricalDistrib(input_size, cfg.theta, cfg.learn_theta)

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
        super(EnergyCategoricalDistrib, self).__init__()
        print(f"input_size: {input_size}")
        if theta is None:
            theta = [1 / prod(input_size)] * prod(input_size)
        assert prod(input_size) == len(theta), "input_size and theta do not match"
        theta_ = torch.Tensor(theta)
        self.theta = nn.parameter.Parameter(theta_, requires_grad=learn_theta)

    def forward(
        self, x: Float[torch.Tensor, "batch_size num_categories"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """Compute the energy of the categorical distribution.

        Args:
            x: Float[torch.Tensor, "batch_size num_categories"], the input of the energy. This is a batch of vectors of size dim.

        Returns:
            Float[torch.Tensor, "batch_size 1"], E(x), the energy of the categorical distribution.
        """
        return (x @ self.theta).reshape(-1, 1)
