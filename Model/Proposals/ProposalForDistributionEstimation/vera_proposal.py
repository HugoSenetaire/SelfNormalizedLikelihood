"""
Defines generator models.
"""

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn

from ...Sampler.hmc_grathwohl import hmc
from ..utils_proposal.network_for_proposal import get_network_for_proposal
from .abstract_proposal import AbstractProposal


def get_vera_hmc(
    input_size,
    dataset,
    cfg,
):
    g = get_network_for_proposal(input_size, cfg)
    return VERAHMCGenerator(input_size, g, cfg.noise_dim, cfg.mcmc_lr)


def get_vera(
    input_size,
    dataset,
    cfg,
):
    g = get_network_for_proposal(input_size, cfg)
    return VERAGenerator(
        input_size, g, cfg.noise_dim, cfg.post_lr, cfg.init_post_logsigma
    )


class VERAHMCGenerator(AbstractProposal):
    """
    VERA Generator with HMC estimator.
    """

    def __init__(self, input_size, g, noise_dim, mcmc_lr=0.02):
        super().__init__(input_size)
        self.input_size = input_size
        self.g = g
        self.logsigma = nn.Parameter(
            (
                torch.ones(
                    1,
                )
                * 0.01
            ).log()
        )
        self.noise_dim = noise_dim
        self.stepsize = nn.Parameter(torch.tensor(1.0 / noise_dim), requires_grad=False)
        self.mcmc_lr = mcmc_lr
        self.ar = 0.0

    def sample_simple(self, nb_sample):
        return self.sample_vera(nb_sample)[0]

    def log_prob_simple(self, x):
        return self.logq_joint(
            x,
            self.sample_vera(x.shape[0], return_mu=True)[1],
        )

    def sample_vera(self, n, requires_grad=False, return_mu=False, return_both=False):
        """sample x, h ~ q(x, h)"""
        h = torch.randn((n, self.noise_dim)).to(next(self.parameters()).device)
        if requires_grad:
            h.requires_grad_()
        x_mu = self.g(h)
        x = x_mu + torch.randn_like(x_mu) * self.logsigma.exp()
        if return_both:
            return x_mu, x, h
        if return_mu:
            return x_mu, h
        else:
            return x, h

    def logq_joint(self, x, h, return_mu=False):
        """
        Join distribution of data and latent.
        """
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        gmu = self.g(h)
        px_given_h = distributions.Normal(gmu, self.logsigma.exp())
        logpx_given_h = px_given_h.log_prob(x).flatten(start_dim=1).sum(1)
        if return_mu:
            return logpx_given_h + logph, gmu
        else:
            return logpx_given_h + logph

    def entropy_obj(
        self,
        x,
        h,
        burn_in=2,
        num_samples_posterior=2,
        return_score=False,
        return_accept=False,
    ):
        """
        Entropy estimator using HMC samples.
        """
        h_given_x, self.ar, self.stepsize.data = hmc.get_gen_posterior_samples(
            netG=self.g,  # function to do HMC on
            x_tilde=x.detach(),  # variable to condition on
            eps_init=h.clone(),  # initialized at stationarity
            sigma=self.logsigma.exp().detach(),
            burn_in=burn_in,
            num_samples_posterior=num_samples_posterior,
            leapfrog_steps=5,
            stepsize=self.stepsize,
            flag_adapt=1,
            hmc_learning_rate=self.mcmc_lr,
            hmc_opt_accept=0.67,
        )  # target acceptance rate, for tuning the LR

        mean_output_summed = torch.zeros_like(x)
        mean_output = self.g(h_given_x)
        for cnt in range(num_samples_posterior):
            mean_output_summed = (
                mean_output_summed
                + mean_output[cnt * x.size(0) : (cnt + 1) * x.size(0)]
            )
        mean_output_summed /= num_samples_posterior

        c = ((x - mean_output_summed) / self.logsigma.exp() ** 2).detach()
        mgn = c.norm(2, 1).mean()
        g_error_entropy = torch.mul(c, x).mean(0).sum()
        if return_score:
            return g_error_entropy, mgn, c
        elif return_accept:
            return g_error_entropy, mgn, acceptRate
        else:
            return g_error_entropy, mgn

    def clamp_sigma(self, sigma, sigma_min=0.01):
        """
        Sigma clamping used for entropy estimator.
        """
        self.logsigma.data.clamp_(np.log(sigma_min), np.log(sigma))


class VERAGenerator(VERAHMCGenerator):
    """
    VERA generator.
    """

    def __init__(self, input_size, g, noise_dim, post_lr=0.001, init_post_logsigma=0.1):
        super().__init__(input_size, g, noise_dim, post_lr)
        self.post_logsigma = nn.Parameter(
            (
                torch.ones(
                    noise_dim,
                )
                * init_post_logsigma
            ).log()
        )
        self.post_optimizer = torch.optim.Adam([self.post_logsigma], lr=post_lr)

    def entropy_obj(
        self, x, h, num_samples_posterior=20, return_score=False, learn_post_sigma=True
    ):
        """
        Entropy objective using variational approximation with importance sampling.
        """
        inf_dist = distributions.Normal(h, self.post_logsigma.detach().exp())
        h_given_x = inf_dist.sample((num_samples_posterior,))

        inf_dist.log_prob(h_given_x).sum(2)
        xr = x.unsqueeze(0).expand(num_samples_posterior, *x.shape)
        xr = xr.reshape(x.shape[0] * num_samples_posterior, *x.shape[1:])
        log_q, mean_output = self.logq_joint(
            xr, h_given_x.view(-1, h.shape[1]), return_mu=True
        )

        mean_output = mean_output.view(num_samples_posterior, *x.shape)
        log_q = log_q.view(num_samples_posterior, x.shape[0])
        w = (log_q - inf_dist.log_prob(h_given_x).sum(2)).softmax(dim=0)
        fvals = (x.unsqueeze(0) - mean_output) / (self.logsigma.exp() ** 2)
        w = w.reshape(
            num_samples_posterior,
            x.shape[0],
            *list(np.ones(len(x.shape[1:]), dtype=int))
        )
        weighted_fvals = (fvals * w).sum(0).detach()
        c = weighted_fvals
        mgn = c.norm(2, 1).mean()
        g_error_entropy = torch.mul(c, x).mean(0).sum()

        post = distributions.Normal(h.detach(), self.post_logsigma.exp())
        h_g_post = post.rsample()
        joint = self.logq_joint(x.detach(), h_g_post)
        post_ent = post.entropy().sum(1)

        elbo = joint + post_ent
        post_loss = -elbo.mean()

        if learn_post_sigma:
            self.post_optimizer.zero_grad()
            post_loss.backward()
            self.post_optimizer.step()

        if return_score:
            return g_error_entropy, mgn, c
        else:
            return g_error_entropy, mgn
