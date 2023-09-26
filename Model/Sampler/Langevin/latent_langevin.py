import numpy as np
import torch
import torch.autograd as autograd
import torch.distributions as dist
import tqdm


def latent_langevin_step(
    z,
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

    z.requires_grad = True

    energy_value = energy(energy.proposal(z))
    z_grad = autograd.grad(
        energy_value.sum(),
        z,
    )[0]
    if clip_max_norm is not None and clip_max_norm != np.inf:
        norm = torch.norm(z_grad.flatten(1), p=2, dim=1, keepdim=True)
        while len(norm.shape) < len(z_grad.shape):
            norm = norm.unsqueeze(-1)
        z_grad = torch.where(
            norm > clip_max_norm, z_grad / norm * clip_max_norm, z_grad
        )

    if clip_max_value is not None and clip_max_value != np.inf:
        z_grad.clamp_(min=-clip_max_value, max=clip_max_value)

    z.requires_grad = False
    noise = torch.randn_like(z) * sigma
    z_step = z - step_size * z_grad + np.sqrt(2 * step_size) * noise

    if clamp_min is not None:
        z_step.clamp_(min=clamp_min)
    if clamp_max is not None:
        z_step.clamp_(max=clamp_max)

    # x_step = proposal(z_step)
    return z_step.detach()


def latent_langevin_sample(
    z_init,
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
    print("Burn in: ", burn_in)
    for k in tqdm.tqdm(range(burn_in)):
        z_init = latent_langevin_step(
            z_init,
            energy,
            step_size,
            sigma,
            clip_max_norm=clip_max_norm,
            clip_max_value=clip_max_value,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    z_samples = []
    for k in tqdm.tqdm(range(num_samples)):
        for t in range(thinning):
            z_init = latent_langevin_step(
                z_init,
                energy,
                step_size,
                sigma,
                clip_max_norm=clip_max_norm,
                clip_max_value=clip_max_value,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
        z_samples.append(z_init)

    z_samples = torch.cat(z_samples, dim=0)

    x_samples = energy.proposal(z_samples)

    return x_samples


class LatentLangevinSampler:
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

    def sample(self, energy_function, z_init=None, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples

        # current_energy_function = lambda x: energy_function(x[0].unsqueeze(0))
        if z_init is None:
            z_init = dist.Normal(
                torch.zeros(self.input_size), torch.ones(self.input_size)
            )(self.num_chains).to(torch.float32)

        langevin_samples = latent_langevin_sample(
            z_init,
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
        x_init = energy_function.proposal(z_init)

        return langevin_samples, x_init
