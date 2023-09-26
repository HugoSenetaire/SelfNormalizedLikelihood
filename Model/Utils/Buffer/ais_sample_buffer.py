
import torch
import torchvision
import wandb

from ...Sampler.Langevin import langevin_step
from .sample_buffer import SampleBuffer


class AIS_Sample_Buffer(SampleBuffer):
    def __init__(self, cfg, buffer_name="ais_sample_buffer"):
        super().__init__(cfg, buffer_name)
        self.buffer_weights = []
        self.batch_of_update = cfg.batch_of_update
        self.store_on_gpu = cfg.store_on_gpu
        self.save_buffer_every = cfg.save_buffer_every

    def populate_buffer(self, ebm,):
        """
        Populate the buffer with samples from the proposal.
        """
        if self.store_on_gpu:
            self.device = next(ebm.parameters()).device
        else :
            self.device = "cpu"
        if hasattr(ebm, "base_dist") and ebm.base_dist is not None:
            use_base_dist = True
            ebm.base_dist.eval()
            to_sample_from = ebm.base_dist
        else:
            ebm.proposal.eval()
            use_base_dist = False
            to_sample_from = ebm.proposal

        for _ in range(self.max_size_buffer):
            with torch.no_grad():
                if use_base_dist:
                    samples = to_sample_from.sample(1, return_log_prob=False)
                    log_prob = torch.zeros(1, device=samples.device, dtype=samples.dtype).reshape(1)
                else:
                    samples, log_prob = to_sample_from.sample(1, return_log_prob=True)

                self.buffer_image.append(samples.detach().to(self.device))
                self.buffer_weights.append(log_prob.detach().to(self.device))
        to_sample_from.train()


    def update_buffer(self, ebm,):
        """
        Update the buffer with samples from the proposal.
        """
        index_update = 0
        while index_update * self.batch_of_update < self.max_size_buffer:
            x_init = torch.cat(self.buffer_image[index_update*self.batch_of_update:(index_update+1)*self.batch_of_update], 0).clone().to(next(ebm.parameters()).device)
            for k in range(self.nb_steps_langevin):
                x_init = langevin_step(
                    x_init=x_init,
                    energy=lambda x: ebm.calculate_energy(x,)[0],
                    step_size=self.step_size_langevin,
                    sigma=self.sigma_langevin,
                    clip_max_norm=self.clip_max_norm,
                    clip_max_value=self.clip_max_value,
                    clamp_max=self.clamp_max,
                    clamp_min=self.clamp_min,
                ).detach()

            log_prob = ebm.calculate_energy(x_init)[0]
            for k in range(index_update, min(index_update+self.batch_of_update, self.max_size_buffer)):
                self.buffer_image[k] = x_init[k-index_update,None,].detach().to(self.device)
                self.buffer_weights[k] = log_prob[k-index_update,None,].detach().to(self.device)
            index_update+=1

    def save_buffer(self, logger, current_step):
        images = torch.cat(self.buffer_image[:64],)
        grid = torchvision.utils.make_grid(images,)
        image = wandb.Image(grid, caption="{}_{}.png".format(self.buffer_name, current_step))
        logger.log({"{}.png".format(self.buffer_name,): image},step=current_step,)



    def get(self, n_samples, device="cuda"):
        index = torch.randint(0, len(self.buffer_image), (n_samples,))
        samples = torch.cat([self.buffer_image[i] for i in index], 0).to(device)
        weights = torch.tensor([self.buffer_weights[i] for i in index]).to(device)
        return samples, weights
    