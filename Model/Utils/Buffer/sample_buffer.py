import torch
import torchvision
import wandb


class SampleBuffer:
    def __init__(self,cfg, buffer_name = "default_buffer"):
        self.buffer_image = []
        self.buffer_ids = []

        self.buffer_name = buffer_name
        self.max_size_buffer = cfg.size_buffer
        self.nb_steps_langevin = cfg.nb_steps_langevin
        self.step_size_langevin = cfg.step_size_langevin
        self.sigma_langevin = cfg.sigma_langevin
        self.clip_max_norm = cfg.clip_max_norm
        self.clip_max_value = cfg.clip_max_value
        self.clamp_max = cfg.clamp_max
        self.clamp_min = cfg.clamp_min


    def __len__(self):
        return len(self.buffer_image)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to("cpu")
        class_ids = class_ids.detach().to("cpu")
        for sample, class_id in zip(samples, class_ids):
            self.buffer_image.append(sample.detach())
            self.buffer_ids.append(class_id)
            if len(self.buffer_image) > self.max_size_buffer:
                self.buffer_image.pop(0)
                self.buffer_ids.pop(0)

    def get(self, n_samples, device="cuda"):
        index = torch.randint(0, len(self.buffer_image), (n_samples,))
        samples = torch.stack([self.buffer_image[i] for i in index], 0).to(device)
        class_ids = torch.tensor([self.buffer_ids[i] for i in index]).to(device)
        return samples, class_ids
    
    def save_buffer(self, logger, current_step):
        images = torch.stack(self.buffer_image[:64],)
        grid = torchvision.utils.make_grid(images,)
        image = wandb.Image(grid, caption="{}_{}.png".format(self.buffer_name, current_step))
        logger.log({"{}.png".format(self.buffer_name,): image},step=current_step,)

