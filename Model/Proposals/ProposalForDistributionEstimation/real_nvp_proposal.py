
from ..real_nvp import RealNVP, RealNVPLoss
from .abstract_proposal import AbstractProposal

import numpy as np
import torch


def get_RealNVPProposal(input_size, dataset, cfg,):
    return RealNVPProposal(input_size=input_size, cfg=cfg,)

class RealNVPProposal(AbstractProposal):
    def __init__(self, input_size, cfg,):
        super(RealNVPProposal, self).__init__(input_size=input_size)
        self.in_channels = self.input_size[0]

        self.num_scales = cfg.real_nvp_num_scales
        self.mid_channels = cfg.real_nvp_mid_channels
        self.num_blocks = cfg.real_nvp_num_blocks
        self.preprocess = cfg.real_nvp_preprocess
        self.k = cfg.real_nvp_k


        self.flow = RealNVP(num_scales=self.num_scales,
                            in_channels=self.in_channels,
                            mid_channels=self.mid_channels,
                            num_blocks=self.num_blocks,
                            pre_process=self.preprocess)

    def log_prob_simple(self, x):
        z, sldj = self.flow(x, reverse=False)
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        return ll

    def sample_simple(self, nb_sample: int = 1):
        self.flow.eval()
        z = torch.randn(nb_sample, *self.input_size, dtype=torch.float32, device=next(self.parameters()).device)
        x, _ = self.flow(z, reverse=True)
        if self.preprocess:
            x = torch.sigmoid(x)
        self.flow.train()
        return x