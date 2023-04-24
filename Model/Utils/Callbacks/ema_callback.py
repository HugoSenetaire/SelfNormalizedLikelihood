from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch

class EMACallback(StochasticWeightAveraging):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
    
    def avg_fn (
        averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
    ) -> torch.FloatTensor:
        e = averaged_model_parameter
        m = model_parameter
        return self.decay * e + (1. - self.decay) * m