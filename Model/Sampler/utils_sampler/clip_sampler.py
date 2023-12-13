
import torch
import numpy as np


def clip_grad(x_grad, clip_max_norm=None, clip_max_value=None):
    if clip_max_norm is not None and clip_max_norm>0. and clip_max_norm != np.inf:
        norm = torch.norm(x_grad.flatten(1), p=2, dim=1, keepdim=True)
        while len(norm.shape) < len(x_grad.shape):
            norm = norm.unsqueeze(-1)
        x_grad = torch.where(norm > clip_max_norm, x_grad/norm * clip_max_norm, x_grad)
    
    if clip_max_value is not None and clip_max_value>0 and clip_max_value != np.inf:
        x_grad.clamp_(min=-clip_max_value, max=clip_max_value)
    
    return x_grad

def clip_data(x_step, clamp_min=None, clamp_max=None):
    if clamp_min is not None:
        x_step.clamp_(min=clamp_min)
    if clamp_max is not None:
        x_step.clamp_(max=clamp_max)

    return x_step