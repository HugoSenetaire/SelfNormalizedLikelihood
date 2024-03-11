import numpy as np
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch

def get_fc_energy(
    input_size,
    cfg,
):
    return fc_energy(
        input_size,
        cfg.dims,
        cfg.activation,
        cfg.last_layer_bias,
    )


def get_fc_energy_sn(input_size, cfg):
    return fc_energy_sn(
        input_size,
        cfg.dims,
        cfg.activation,
        cfg.last_layer_bias,
    )


def get_fc_energy_sn_miniboone(input_size, cfg):
    return fc_energy_sn_miniboone(
        input_size,
        cfg.dims,
        cfg.activation,
        cfg.last_layer_bias,
    )


class fc_energy(nn.Module):
    def __init__(
        self,
        input_size=(1, 10),
        dims=[100, 100, 100],
        activation=None,
        last_layer_bias=True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear = [
            nn.Linear(np.prod(input_size), dims[0]),
        ]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(nn.Linear(dim_in, dim_out))
        self.linear.extend(
            [
                nn.ReLU(),
            ]
        )
        self.linear = nn.Sequential(*self.linear)
        self.last_layer = nn.Linear(dims[-1], 1, bias=last_layer_bias)
        self.activation = None

    def forward(self, x):
        x = x.flatten(1)
        out = self.linear(x)
        out = self.last_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        return out.reshape(-1, 1)


class fc_energy_sn(nn.Module):
    def __init__(
        self,
        input_size=(1, 10),
        dims=[100, 100, 100],
        activation=None,
        last_layer_bias=False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear = [
            spectral_norm(nn.Linear(np.prod(input_size), dims[0])),
        ]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(spectral_norm(nn.Linear(dim_in, dim_out)))
        self.linear.extend(
            [
                nn.ReLU(),
            ]
        )
        self.linear = nn.Sequential(*self.linear)
        # self.last_layer = spectral_norm(nn.Linear(dims[-1], 1, bias = last_layer_bias))
        self.last_layer = nn.Linear(dims[-1], 1, bias=last_layer_bias)
        self.activation = None

    def forward(self, x):
        x = x.flatten(1)
        out = self.linear(x)
        out = self.last_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        return out.reshape(-1, 1)


class fc_energy_sn_miniboone(nn.Module):
    def __init__(
        self,
        input_size=(1, 10),
        dims=[100, 100, 100],
        activation=None,
        last_layer_bias=True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear = [
            spectral_norm(nn.Linear(np.prod(input_size), dims[0])),
        ]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(spectral_norm(nn.Linear(dim_in, dim_out)))
        self.linear.extend(
            [
                nn.ReLU(),
            ]
        )
        self.linear = nn.Sequential(*self.linear)
        # self.last_layer = nn.Linear(dims[-1], 1, bias = last_layer_bias)
        self.last_layer = spectral_norm(nn.Linear(dims[-1], 1, bias=last_layer_bias))
        self.activation = None

    def forward(self, x):
        x = x.flatten(1)
        out = self.linear(x)
        out = self.last_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        # out = 10* torch.nn.functional.tanh(out)
        return out.reshape(-1, 1)
