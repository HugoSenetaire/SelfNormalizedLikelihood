import math

import numpy as np
import torch
import torch.nn as nn

from ..pytorch_flows import FlowSequential, get_flow
from .abstract_proposal import AbstractProposal


def get_PytorchFlowsProposal(
    input_size,
    dataset,
    cfg,
):
    proposal = PytorchFlowsProposal(
        input_size=input_size,
        cfg=cfg,
    )

    x = torch.stack([dataset.__getitem__(k)["data"] for k in range(64)]).flatten(1)
    if torch.cuda.is_available():
        x = x.cuda()
        proposal = proposal.cuda()
    aux = proposal.log_prob_simple(x)
    return proposal


class PytorchFlowsProposal(AbstractProposal):
    def __init__(
        self,
        input_size,
        cfg,
    ):
        super(PytorchFlowsProposal, self).__init__(input_size=input_size)
        self.dim = np.prod(self.input_size)
        self.flow_name = cfg.pytorch_flow_name
        self.hidden_dim = cfg.pytorch_flow_hidden_dim
        self.num_blocks = cfg.pytorch_flow_num_blocks
        self.num_cond_inputs = None
        self.act = cfg.pytorch_flow_act

        self.modules = get_flow(
            self.flow_name,
            self.dim,
            self.hidden_dim,
            self.num_cond_inputs,
            self.num_blocks,
            act="relu",
        )

        self.flow = FlowSequential(*self.modules)
        self.flow.num_inputs = self.dim

        for module in self.flow.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.fill_(0)

    def log_prob_simple(self, x):
        x = x.flatten(1)
        u, log_det = self.flow.forward(x)
        negloglik_loss = 0.5 * (u**2).sum(dim=1)
        negloglik_loss += 0.5 * x.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det.reshape(negloglik_loss.shape)
        return -negloglik_loss
        # negloglik_loss = torch.mean(negloglik_loss)

    def sample_simple(self, nb_sample: int = 1):
        x = self.flow.sample(nb_sample)
        x = x.reshape(nb_sample, *self.input_size)
        return x
