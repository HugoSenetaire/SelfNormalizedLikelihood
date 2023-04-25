
import torch 
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
from tqdm import tqdm

class HMCSampler(nn.Module):
    def __init__(self, f, eps, n_steps, init_sample, scale_diag=None, covariance_matrix=None, device=None):
        super(HMCSampler, self).__init__()
        self.init_sample = init_sample
        self.f = f
        self.eps = eps
        if scale_diag is not None:
            self.p_dist = distributions.Normal(loc=0., scale=scale_diag.to(device))
        else:
            self.p_dist = distributions.MultivariateNormal(loc=torch.zeros_like(covariance_matrix)[:, 0].to(device),
                                                           covariance_matrix=covariance_matrix.to(device))
        self.n_steps = n_steps
        self.device = device
        self._accept = 0.

    def _grad(self, z):
        return torch.autograd.grad(-self.f(z).sum(), z, create_graph=True)[0]

    def _kinetic_energy(self, p):
        return -self.p_dist.log_prob(p).view(p.size(0), -1).sum(dim=-1)

    def _energy(self, x, p):
        k = self._kinetic_energy(p)
        pot = -self.f(x)
        return k + pot

    def initialize(self):
        x = self.init_sample()
        return x

    def _proposal(self, x, p):
        g = self._grad(x.requires_grad_())
        xnew = x
        gnew = g
        for _ in range(self.n_steps):
            p = p - self.eps * gnew / 2.
            xnew = (xnew + self.eps * p)
            gnew = self._grad(xnew.requires_grad_())
            xnew = xnew#.detach()
            p = p - self.eps * gnew / 2.
        return xnew, p

    def step(self, x):
        p = self.p_dist.sample_n(x.size(0))
        pc = torch.clone(p)
        xnew, pnew = self._proposal(x, p)
        assert (p == pc).all().float().item() == 1.0
        Hnew = self._energy(xnew, pnew)
        Hold = self._energy(x, p)

        diff = Hold - Hnew
        shape = [i if no == 0 else 1 for (no, i) in enumerate(x.shape)]

        accept = (diff.exp() >= torch.rand_like(diff)).to(x).view(*shape)
        x = accept * xnew + (1. - accept) * x
        self._accept = accept.mean()
        return x.detach()

    def sample(self, n_steps):
        x = self.initialize().to(self.device)
        t = tqdm(range(n_steps))
        accepts = []
        for _ in t:
            x = self.step(x)
            t.set_description("Acceptance Rate: {}".format(self._accept))
            accepts.append(self._accept.item())
        accepts = np.mean(accepts)
        if accepts < .4:
            self.eps *= .67
            print("Decreasing epsilon to {}".format(self.eps))
        elif accepts > .9:
            self.eps *= 1.33
            print("Increasing epsilon to {}".format(self.eps))
        return x