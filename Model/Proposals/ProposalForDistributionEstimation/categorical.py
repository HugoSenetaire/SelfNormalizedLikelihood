from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch.distributions import categorical
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from .abstract_proposal import AbstractProposal


def get_Categorical(input_size, dataset, cfg):
    return Categorical(input_size, dataset)

class Categorical(AbstractProposal):
    """Module that implements a uniform categorical distribution"""

    def __init__(self, input_size: Tuple[int], dataset: Dataset) -> None:
        super(Categorical, self).__init__(input_size=input_size)
        self.num_categories = dataset.num_categories
        self.logit_parameters = Parameter(torch.ones(self.num_categories))

    def sample_simple(self, nb_sample: int = 1):
        samples = torch.Tensor(
            categorical.Categorical(self.logit_parameters).sample((nb_sample,))
        ).type("torch.LongTensor")
        samples_one_hot = F.one_hot(samples, num_classes=self.num_categories).type(
            "torch.FloatTensor"
        )

        return samples_one_hot

    def log_prob_simple(self, x: Float[torch.Tensor, "batch_size"]):
        return (
            categorical.Categorical(self.logit_parameters).log_prob(x).flatten(1).sum(1)
        )
