import numpy as np
import torch
import torch.autograd as autograd
import torch.distributions as dist
import tqdm
from ..utils_sampler.clip_sampler import clip_grad, clip_data


def langevin_step(x_init, energy, step_size, sigma,  clip_max_norm=None, clip_max_value=None, clamp_min=None, clamp_max=None):
    """
    Performs a single step of the Langevin algorithm.
    """

    x_init.requires_grad = True

    energy_value = energy(x_init)
    x_grad = autograd.grad(energy_value.sum(), x_init, )[0]
    x_grad = clip_grad(x_grad, clip_max_norm=clip_max_norm, clip_max_value=clip_max_value)


    x_init.requires_grad = False
    noise = torch.randn_like(x_init) * sigma
    x_step = x_init - step_size * x_grad + np.sqrt(2*step_size)*noise
    x_step = clip_data(x_step, clamp_min=clamp_min, clamp_max=clamp_max)
   

    return x_step.detach()


def langevin_sample(x_init, energy, step_size, sigma, num_samples, clip_max_norm = None, clip_max_value = None, clamp_min = None, clamp_max = None, burn_in=0, thinning=0):
    """
    Performs a single step of the Langevin algorithm.
    """
    print("Langevin sampling, burn in: {}, thinning: {}, num samples: {}".format(burn_in, thinning, num_samples))
    for k in tqdm.tqdm(range(burn_in)):
        x_init = langevin_step(x_init, energy, step_size, sigma, clip_max_norm=clip_max_norm, clip_max_value=clip_max_value, clamp_min=clamp_min, clamp_max=clamp_max)
    x_samples = []
    for k in tqdm.tqdm(range(num_samples)):
        for t in range(thinning):
            x_init = langevin_step(x_init, energy, step_size, sigma, clip_max_norm=clip_max_norm, clip_max_value=clip_max_value, clamp_min=clamp_min, clamp_max=clamp_max)
        x_samples.append(x_init)

    x_samples = torch.cat(x_samples, dim=0)

    return x_samples






class LangevinSampler:
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
        
        
        langevin_samples = langevin_sample(
            x_init,
            energy_function,
            step_size=self.step_size,
            sigma = self.sigma,
            num_samples=num_samples,
            burn_in=self.warmup_steps,
            thinning=self.thinning,
            clip_max_norm=self.clip_max_norm,
            clip_max_value=self.clip_max_value,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
        )
     
        return langevin_samples, x_init
