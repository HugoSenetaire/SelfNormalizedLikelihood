import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from .abstract_proposal import AbstractProposal


def get_NeuralODEProposal(input_size, cfg,  f_theta, base_dist = None,):
    return NeuralODEProposal(
        input_size=input_size,
        f_theta=f_theta,
        base_dist=base_dist,
        cfg=cfg,
    )


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.0
    nb_dims = torch.flatten(z, start_dim=1).shape[1]
    flat_f = torch.flatten(f, start_dim=1)
    for i in range(nb_dims):  # For each dimension
        elmnt = (
            torch.autograd.grad(flat_f[:, i].sum(), z, create_graph=True)[0].flatten(1)
            .contiguous()[:, i]
            .contiguous()
        )
        sum_diag += elmnt
    flat_diag = sum_diag
    return flat_diag.contiguous()


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def m_score(f_theta, base_dist, x, **unused_kwargs):
    energy = f_theta(x)  # should be a scalar
    if base_dist is not None:
        energy -= base_dist.log_prob(x)
    score = torch.autograd.grad(energy.sum(), x, retain_graph=True, create_graph=True)[
        0
    ]
    return -score


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


class Func(nn.Module):
    def __init__(
        self,
        f_theta: nn.Module,
        base_dist: nn.Module,
        divergence_fn: str = "approximate",
    ):
        super(Func, self).__init__()
        self.f_theta = f_theta
        self.base_dist = base_dist
        self.divergence_fn = divergence_fn
        if self.divergence_fn == "approximate":
            self.div_fn = divergence_approx
        elif self.divergence_fn == "exact":
            self.div_fn = trace_df_dz
        else:
            raise ValueError("divergence_fn should be 'approximate' or 'exact', got {}".format(divergence_fn))

    def forward(self, t, states):
        x, logp_x = states
        x.requires_grad_(True)
        dx_dt = m_score(self.f_theta, self.base_dist, x)
        if self.divergence_fn == "approximate":
            e = sample_rademacher_like(x)
            dlog_x_dt = -self.div_fn(dx_dt, x, e)
        elif self.divergence_fn == "exact":
            e = None
            dlog_x_dt = -self.div_fn(dx_dt, x,)
        else:
            raise ValueError("divergence_fn should be 'approximate' or 'exact', got {}".format(self.divergence_fn))
        return dx_dt, dlog_x_dt


class NeuralODEProposal(AbstractProposal):
    def __init__(
        self,
        input_size,
        cfg,
        f_theta,
        base_dist=None,
    ):
        super(NeuralODEProposal, self).__init__(input_size=input_size)
        self.f_theta = f_theta
        self.base_dist = base_dist
        self.divergence_fn = cfg.cnf_divergence_fn
        self.method = cfg.cnf_method
        self.nb_solver_step = cfg.cnf_nb_solver_step
        self.T = cfg.cnf_T
        self.fake_param = nn.Parameter(torch.tensor(0.0))
        self.func = Func(
            f_theta=self.f_theta, base_dist=self.base_dist, divergence_fn="exact"
        )
        flat_input_size = np.prod(input_size)
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(flat_input_size), torch.eye(flat_input_size)
        )

    def log_prob_simple(self, x_1):
        logp_diff_t0 = torch.zeros(x_1.shape[0], 1).type(torch.float32)
        x_t, logp_x_t = odeint(
            self.func,
            (x_1, logp_diff_t0),
            torch.tensor(np.linspace(self.T, 0, self.nb_solver_step)),
            method="euler",
        )
        # x_t (time_steps, batch_size, *input_size)
        x_0 = torch.flatten(x_t[-1], start_dim=1)
        logp_x_0 = logp_x_t[-1]
        prior_log_prob = self.prior.log_prob(x_0)
        return prior_log_prob + logp_x_0

    def sample_simple(self, nb_sample: int = 1):
        x_0 = self.prior.sample((nb_sample,)).to(self.fake_param.device).reshape(nb_sample, *self.input_size)
        logp_diff_t0 = torch.zeros(nb_sample, 1).to(dtype=torch.float32, device=self.fake_param.device)
        x_t, logp_x_t = odeint(
            self.func,
            (x_0, logp_diff_t0),
            torch.tensor(np.linspace(0, self.T, self.nb_solver_step)),
            method="euler",
        )
        return x_t[-1], logp_x_t[-1]

    def sample(self, nb_sample, return_log_prob=False):
        """
        Sample from the proposal.
        """
        samples, log_prob = self.sample_simple(nb_sample=nb_sample)
        if return_log_prob:
            log_prob = self.log_prob_simple(samples)
            return samples.reshape(nb_sample, *self.input_size).detach(), log_prob.reshape(nb_sample, 1).detach()
        else:
            return samples.reshape(nb_sample, *self.input_size).detach()


####################################################################################################


# class Energy(nn.Module):
#     def __init__(self):
#         super(Energy, self).__init__()
#         self.theta = nn.Parameter(torch.tensor(3.0))

#     def forward(self, x):
#         # print(f"x.shape = {x.shape} in energy")
#         return torch.flatten(self.theta * x**2, start_dim=1).sum(dim=1)


# class EnergyMultivariateNormal(nn.Module):
#     def __init__(self):
#         super(EnergyMultivariateNormal, self).__init__()
#         self.mean = nn.Parameter(torch.tensor([[2.0, 2.0]]))
#         self.inv_cov = (1 / 5) * nn.Parameter(torch.tensor([[4.0, -3.0], [-1.0, 2.0]]))

#     def forward(self, x):
#         x = x - self.mean
#         x_inv_cov_x = torch.einsum("bi, ij, bj -> b", x, self.inv_cov, x)
#         return 0.5 * x_inv_cov_x


# func = Func(EnergyMultivariateNormal(), "exact")

# p_x0 = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

# x_0 = p_x0.sample((3,))  # Sample 2 points from p_x0
# # print(f"x_0.shape = {x_0.shape}")
# # x_0 = x_0.reshape(2, 1)

# logp_diff_t0 = torch.zeros(3, 1).type(torch.float32)

# x_t, logp_x_t = odeint(
#     func, (x_0, logp_diff_t0), torch.tensor(np.linspace(0, 1, 10)), method="euler"
# )

# print(x_t.shape)
# print(logp_x_t.shape)
# print("DONE")
