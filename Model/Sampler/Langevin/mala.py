import numpy as np
import torch
import torch.autograd as autograd
import torch.distributions as dist
import tqdm

from ..utils_sampler.clip_sampler import clip_grad, clip_data


def langevin_mala_step(
    x_init,
    energy,
    step_size,
    sigma,
    clip_max_norm=None,
    clip_max_value=None,
    clamp_min=None,
    clamp_max=None,
):
    """
    Performs a single step of the Langevin algorithm.
    """

    effective_std = (2 * step_size) ** 0.5 * sigma

    x_init.requires_grad = True
    energy_x = energy(x_init)
    x_grad = autograd.grad(energy_x.sum(),x_init,)[0]
    x_grad = clip_grad(x_grad, clip_max_norm=clip_max_norm, clip_max_value=clip_max_value)


    x_mu = x_init - step_size * x_grad
    noise = torch.randn_like(x_init) * effective_std
    y = x_mu + noise

    y = y.detach()
    y.requires_grad = True
    energy_y = energy(y)

    log_prob_forward = dist.Normal(x_mu, effective_std).log_prob(y).flatten(start_dim=1).sum(1) - energy_x

    y_grad = autograd.grad(energy_y.sum(),y,)[0]
    y_grad = clip_grad(y_grad, clip_max_norm=clip_max_norm, clip_max_value=clip_max_value)
    y_mu = y - step_size * y_grad
    log_prob_backward = dist.Normal(y_mu, effective_std).log_prob(x_init).flatten(start_dim=1).sum(1) - energy_y

    log_prob_accept = log_prob_backward - log_prob_forward
    p_accept = log_prob_accept.exp()
    accept = (torch.rand_like(p_accept) < p_accept).float().reshape(-1, *[1 for _ in range(len(x_init.shape) - 1)])
    y = accept * y + (1 - accept) * x_init
    y = y.detach()
    y = clip_data(y, clamp_min=clamp_min, clamp_max=clamp_max)

    return y, accept.mean()


def langevin_mala_sample(
    x_init,
    energy,
    step_size,
    sigma,
    num_samples,
    clip_max_norm=None,
    clip_max_value=None,
    clamp_min=None,
    clamp_max=None,
    burn_in=0,
    thinning=0,
):
    """
    Performs a single step of the Langevin algorithm.
    """
    
    iter_burn_in = range(burn_in)
    for k in iter_burn_in:
        x_init, accept = langevin_mala_step(
            x_init,
            energy,
            step_size,
            sigma,
            clip_max_norm=clip_max_norm,
            clip_max_value=clip_max_value,
            clamp_max=clamp_max,
            clamp_min=clamp_min,
        )

    x_samples = []
    iter_sample =range(num_samples)

    for k in iter_sample :
        for t in range(thinning):
            x_init, accept = langevin_mala_step(
                x_init,
                energy,
                step_size,
                sigma,
                clip_max_norm=clip_max_norm,
                clip_max_value=clip_max_value,
                clamp_max=clamp_max,
                clamp_min=clamp_min,
            )
        x_samples.append(x_init)

    x_samples = torch.cat(x_samples, dim=0)

    return x_samples


class MetropolisAdjustedLangevinSampler:
    def __init__(
        self,
        input_size=(1, 2),
        num_chains=10,
        num_samples=100,
        warmup_steps=100,
        thinning=10,
        step_size=1e-2,
        sigma=1e-2,
        clip_max_norm=None,
        clip_max_value=None,
        clamp_min=None,
        clamp_max=None,
        **kwargs,
    ):
        self.input_size = input_size
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.sigma = sigma
        self.num_chains = num_chains
        self.thinning = thinning
        self.clip_max_norm = clip_max_norm
        self.clip_max_value = clip_max_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def sample(self, energy_function, x_init=None, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples

        # current_energy_function = lambda x: energy_function(x[0].unsqueeze(0))
        if x_init is None:
            x_init = dist.Normal(
                torch.zeros(self.input_size), torch.ones(self.input_size)
            )(self.num_chains).to(torch.float32)

        langevin_samples = langevin_mala_sample(
            x_init,
            energy_function,
            step_size=self.step_size,
            sigma=self.sigma,
            num_samples=num_samples,
            burn_in=self.warmup_steps,
            thinning=self.thinning,
            clip_max_norm=self.clip_max_norm,
            clip_max_value=self.clip_max_value,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
        )

        return langevin_samples, x_init
