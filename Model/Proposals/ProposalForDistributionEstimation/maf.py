
from ..maf import MAF
from .abstract_proposal import AbstractProposal

import numpy as np
import torch
import math


def get_MAFProposal(input_size, dataset, cfg,):
    return MAFProposal(input_size=input_size, cfg=cfg,)

class MAFProposal(AbstractProposal):
    def __init__(self, input_size, cfg,):
        super(MAFProposal, self).__init__(input_size=input_size)
        self.dim = np.prod(self.input_size)
        self.hidden_dim = cfg.maf_hidden_dim
        self.num_blocks = cfg.maf_num_blocks
        self.use_reverse = cfg.maf_use_reverse
        self.input_size = input_size
        self.flow = MAF(self.dim, self.num_blocks, self.hidden_dim, use_reverse=self.use_reverse)

    def log_prob_simple(self, x):
        x = x.flatten(1)
        u, log_det = self.flow.forward(x)

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * x.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        return -negloglik_loss
        # negloglik_loss = torch.mean(negloglik_loss)

    def sample_simple(self, nb_sample: int = 1):
        z = torch.randn(nb_sample, *self.input_size, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device).flatten(1)
        x, _ = self.flow.backward(z)
        x = x.reshape(-1, *self.input_size)
        return x