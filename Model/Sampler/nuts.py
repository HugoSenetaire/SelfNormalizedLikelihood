
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch

def nuts_sampler(energy_function, proposal = None, input_size = (1, 2), num_chains= 10, num_samples = 10, warmup_steps = 100):

    if proposal is None:
        x_init = dist.Normal(torch.zeros(input_size),torch.ones(input_size))(num_chains)
    else :
        x_init = proposal.sample(num_chains).to(torch.float32)
    hmc_kernel = NUTS(potential_fn = energy_function, adapt_step_size=True, )
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params = {0:x_init}, num_chains=num_chains)
    mcmc.run()
    return mcmc.get_samples()