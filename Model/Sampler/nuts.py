
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch



class NutsSampler():
    def __init__(self,  input_size = (1, 2), num_chains= 10, num_samples = 100, warmup_steps = 100, thinning = 10, **kwargs):
        print(kwargs)
        self.input_size = input_size
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.thinning = thinning

    def sample(self, energy_function, proposal = None, num_samples = None):
        if num_samples is None :
            num_samples = self.num_samples

        # current_energy_function = lambda x: energy_function(x[0].unsqueeze(0))
        if proposal is None:
            x_init = dist.Normal(torch.zeros(self.input_size),torch.ones(self.input_size))(self.num_chains).to(torch.float32)
        else :
            x_init = proposal.sample(self.num_chains).to(torch.float32)
        hmc_kernel = NUTS(potential_fn = energy_function, adapt_step_size=True, )

        samples = []
        for x_init_i in x_init :
            mcmc = MCMC(hmc_kernel, num_samples=num_samples*self.thinning, warmup_steps=self.warmup_steps, initial_params = {0:x_init_i.unsqueeze(0)}, num_chains=1)
            mcmc.run()
            samples.append(mcmc.get_samples()[0].clone().detach().reshape(self.num_samples, self.thinning, 1, *self.input_size)[:,0]) # 0 is because I have defined initial params as 0

        # mcmc = MCMC(hmc_kernel, num_samples=num_samples*self.thinning, warmup_steps=self.warmup_steps, initial_params = {0:x_init}, num_chains=self.num_chains)
        # mcmc.run()
        # samples = 
        # samples = mcmc.get_samples()[0] # 0 is because I have defined initial params as 0
        # samples = samples.reshape(num_samples, self.thinning,self.num_chains,*self.input_size)[:,0].flatten(0,1)
        samples = torch.cat(samples, dim=1).reshape(self.num_samples * self.num_chains, *self.input_size)
        return samples, x_init
    

