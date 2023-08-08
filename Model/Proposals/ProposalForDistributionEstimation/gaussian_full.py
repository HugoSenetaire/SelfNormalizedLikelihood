import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from .abstract_proposal import AbstractProposal


def get_GaussianFull(
    input_size,
    dataset,
    cfg,
):
    return GaussianFull(
        input_size,
        dataset,
        cfg.mean,
        cfg.std,
        cfg.nb_sample_estimate,
        cfg.std_multiplier,
    )


class GaussianFull(AbstractProposal):
    def __init__(
        self,
        input_size,
        dataset,
        mean="dataset",
        std="dataset",
        nb_sample_estimate=50000,
        std_multiplier=1,
        **kwargs
    ) -> None:
        super().__init__(input_size=input_size)
        print("Init Standard Gaussian...")
        data = self.get_data(dataset, nb_sample_estimate).flatten(1).numpy()

        m, s = fit_gaussian(data)
        self.mean = torch.nn.Parameter(torch.tensor(m).float())
        print(self.mean.shape)
        self.s = torch.nn.Parameter(torch.tensor(s + 1e-8).float())
        self.input_size = input_size

    def sample_simple(self, nb_sample=1):
        self.distribution = dist.MultivariateNormal(self.mean, covariance_matrix=self.s)
        samples = self.distribution.sample((nb_sample,))
        samples = samples.reshape(-1, *self.input_size)
        return samples

    def log_prob_simple(self, x):
        self.distribution = dist.MultivariateNormal(self.mean, covariance_matrix=self.s)
        x = x.flatten(1)
        return self.distribution.log_prob(x)


def fit_gaussian(x, w=None):
    """Fits and returns a gaussian to a (possibly weighted) dataset using maximum likelihood."""

    if w is None:
        m = np.mean(x, axis=0)
        xm = x - m
        S = np.dot(xm.T, xm) / x.shape[0]

    else:
        m = np.dot(w, x)
        S = np.dot(x.T * w, x) - np.outer(m, m)

    return m, S
