import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np

def wgan_gradient_penalty(ebm, x, x_gen,):
    batch_size = x.size()[0]
    min_data_len = min(batch_size,x_gen.size()[0])


    # Calculate interpolation
    epsilon = torch.rand(min_data_len, device=x.device)
    for _ in range(len(x.shape) - 1):
        epsilon = epsilon.unsqueeze(-1)
    epsilon = epsilon.expand(min_data_len, *x.shape[1:])
    epsilon = epsilon.to(x.device)


    interpolated = epsilon*x.data[:min_data_len] + (1-epsilon)*x_gen.data[:min_data_len]
    interpolated = Variable(interpolated, requires_grad=True)
    # interpolated = interpolated.detach()
    # interpolated.requires_grad_(True)

    # Calculate probability of interpolated examples
    prob_interpolated = ebm.f_theta(interpolated).flatten(1).sum(1)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(x.device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(min_data_len, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sum(gradients ** 2, dim=1).mean()
    return gradients_norm
