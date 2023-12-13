import itertools

import torch
import torch.distributions as distributions
import torch.nn as nn


class ExplicitBias(nn.Module):
    """
    Used to handle the explicit bias of the EBM model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.explicit_bias = torch.nn.parameter.Parameter(
            torch.zeros(1), requires_grad=True
        )

    def forward(self, x):
        return self.explicit_bias


class ImportanceWeightedEBM(nn.Module):
    """
    Combine f_theta, bias, proposal and base distribution to form an EBM.

    Attributes :
    ------------
    f_theta : torch.nn.Module
        The neural network function of the EBM.
    proposal : torch.nn.Module
        The proposal distribution of the EBM, as implemented in ../Proposal
    base_dist : torch.nn.Module
        The base distribution of the EBM, simply requires a log_prob function.
    explicit_bias : bool
        Whether to use explicit bias or not, if yes, the bias is stored in self.explicit_bias.explicit_bias
    nb_sample_init_bias : int
        The number of samples to use to estimate the explicit bias of the EBM either at the beginning of training.

    Methods :
    ---------
    sample(torch.tensor/int : nb_sample = 1) -> torch.tensor (shape : (nb_sample, *self.energy.input_dim))
        Sample from the proposal distribution.
    calculate_energy(torch.tensor : x, bool : use_base_dist = True) -> torch.tensor (shape : (x.shape[0], 1))
        Calculate the energy of x with the energy function.
    estimate_log_z(torch.tensor : x, int : nb_sample = 1000) -> torch.tensor (shape : (1,))
        Estimate the log-normalization of the ebm using the proposal.
    forward(torch.tensor : x) -> torch.tensor (shape : (x.shape[0], 1))
        Forward function of the model giving the full energy, used to pickle the model and multiprocess HMM.
    """

    def __init__(
        self,
        f_theta,
        proposal,
        base_dist,
        explicit_bias,
        nb_sample_init_bias=1024,
    ):
        super(ImportanceWeightedEBM, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.f_theta = f_theta.to(device)
        self.proposal = proposal.to(device)
        self.nb_sample_init_bias = nb_sample_init_bias
        self.base_dist = base_dist.to(device)
        self.explicit_bias = explicit_bias.to(device)

        # If we use explicit bias, set it to a first estimation of the normalization constant.
        if hasattr(self.explicit_bias, "bias"):
            log_z_estimate, dic = self.estimate_log_z(torch.zeros(1,dtype=torch.float32,).to(device),nb_sample=self.nb_sample_init_bias,)
            self.explicit_bias.bias.data = log_z_estimate

    def sample(self, nb_sample=1, return_log_prob=False):
        """
        Samples from the proposal distribution.
        """
        return self.proposal.sample(nb_sample, return_log_prob=return_log_prob)

    def calculate_energy(self, x, use_base_dist=True):
        """
        Calculate energy of x with the energy function.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input to the energy function.
        use_base_dist : bool
            Whether to use the base distribution or not. In some cases, it's not necessary to calculate it (eg, when proposal = base distribution)

        Returns :
        ---------
        current_energy : torch.tensor (shape : (x.shape[0], 1))
            The energy of x.
        dic_output : dict
            A dictionary containing the different components of the energy
            (ie, f_theta of the data, log-prob of the data under the base distribution,
            explicit bias contribution, complete energy of the data)
        """
        dic_output = {}

        # Get the energy of the samples without base distribution
        f_theta_no_bias = self.f_theta(x).flatten()
        if (
            self.training
            and self.proposal is not None
            and hasattr(
                self.proposal,
                "set_x",
            )
        ):
            # During training, if the proposal is adapative to the batch, set the x of the proposal to the current batch.
            self.proposal.set_x(x)

        # Add the explicit bias contribution to the energy
        f_theta = self.explicit_bias.add_bias(f_theta_no_bias).flatten()

        # Add the base distribution contribution to the energy
        if use_base_dist:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            base_dist_log_prob = self.base_dist.log_prob(x).flatten()
        else:
            base_dist_log_prob = torch.zeros_like(f_theta)

        # Complete energy :
        current_energy = f_theta - base_dist_log_prob

        dic_output["f_theta_no_bias_on_data"] = f_theta_no_bias.detach()
        dic_output["f_theta_on_data"] = f_theta.detach()
        dic_output["explicit_bias_data"] = self.explicit_bias.bias.data.detach()
        dic_output["base_dist_log_likelihood_on_data"] = base_dist_log_prob.detach()
        dic_output["energy_on_data"] = current_energy.detach()

        return current_energy, dic_output

    def estimate_log_z(
        self,
        x,
        nb_sample=1000,
        detach_sample=True,
        detach_base_dist=False,
        requires_grad=False,
        return_samples=False,
        noise_annealing=0.0,
        force_calculation=False,
    ):
        """
        Estimate the log-normalization of the ebm using the proposal.

        Parameters :
        ------------
        x : torch.tensor (shape : (x.shape[0], *input_dim))
            The input to the energy function. Not really useful here.

        nb_sample : int
            The number of samples to use to estimate the log-normalization.

        Returns :
        ---------
        log_z_estimate : torch.tensor (shape : (1,))
            The log-normalization estimate.
        dic_output : dict
            A dictionary containing the different components of the log-normalization estimate
            (ie, f_theta of the samples, log-prob of the samples under the base distribution,
            log-prob of the samples under the proposal, auxilary prob of the samples, log-normalization estimate)

        """
        dic_output = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)

        samples_proposal, samples_proposal_log_prob = self.sample(nb_sample, return_log_prob=True)
        samples_proposal = samples_proposal.to(x.device, x.dtype)
        samples_proposal_log_prob = samples_proposal_log_prob.to(x.device, x.dtype)
        if detach_sample:
            samples_proposal = samples_proposal.detach()
        if requires_grad:
            samples_proposal.requires_grad = True

        if noise_annealing > 1e-4:
            epsilon = torch.rand_like(samples_proposal).to(x.device, x.dtype)
            noise_to_add = epsilon * noise_annealing
            log_prob_noise = torch.distributions.Normal(0, 1).log_prob(epsilon).reshape((samples_proposal.shape[0], -1)).sum(-1)
            samples_proposal_noisy = samples_proposal + noise_to_add
        else:
            log_prob_noise = torch.zeros((nb_sample,)).to(x.device, x.dtype)
            samples_proposal_noisy = samples_proposal

        # Get the energy of the samples_proposal without base distribution
        f_theta_proposal_without_bias = self.f_theta(samples_proposal_noisy).flatten()

        # if self.explicit_bias:
        f_theta_proposal = self.explicit_bias.add_bias(f_theta_proposal_without_bias).flatten()

        # Add the base distribution and proposal contributions to the energy if they are different
        # if self.base_dist != self.proposal or noise_annealing > 1e-4:
        base_dist_log_prob = self.base_dist.log_prob(samples_proposal_noisy).view(samples_proposal.size(0), -1).sum(1)

        energy_samples = f_theta_proposal - base_dist_log_prob
        if detach_base_dist:
            base_dist_log_prob = base_dist_log_prob.detach()
        aux_prob = base_dist_log_prob - samples_proposal_log_prob - log_prob_noise

        if self.base_dist == self.proposal and not force_calculation:
            log_z_estimate = -f_theta_proposal.flatten() - log_prob_noise.flatten()
        else:
            log_z_estimate = (-f_theta_proposal + aux_prob).flatten()

        ESS_estimate_num = log_z_estimate.logsumexp(0) * 2
        ESS_estimate_denom = (2 * log_z_estimate).logsumexp(0)
        ESS_estimate = (ESS_estimate_num - ESS_estimate_denom).exp()
        log_z_estimate = torch.logsumexp(log_z_estimate, dim=0) - torch.log(torch.tensor(nb_sample, dtype=x.dtype, device=x.device))

        dic_output.update(
            {
                "z_estimation/f_theta_no_bias_on_sample_proposal": f_theta_proposal_without_bias.detach(),
                "z_estimation/f_theta_on_sample_proposal": f_theta_proposal.detach(),
                "z_estimation/base_dist_loglikelihood_on_sample_proposal": base_dist_log_prob.detach(),
                "z_estimation/proposal_loglikelihood_on_sample_proposal": samples_proposal_log_prob.detach(),
                "z_estimation/aux_prob_on_sample_proposal": aux_prob.detach(),
                "z_estimation/log_prob_noise": log_prob_noise.detach(),
                "z_estimation/log_z_estimate": log_z_estimate.detach(),
                "z_estimation/ESS_estimate": ESS_estimate.detach(),
            }
        )
        to_return = [log_z_estimate, dic_output]
        if return_samples:
            to_return.extend([energy_samples, samples_proposal, samples_proposal_noisy])
       
        return to_return

    def forward(self, x):
        """
        Had to create a separate forward function in order to pickle the model and multiprocess HMM.

        Parameters :
        ------------
        x : dict(0:torch.tensor (shape : (x.shape[0], *input_dim)))
            The input to the energy function. Use in a dic with 0 as keys to be compatible with the NUTS sampler.

        Returns :
        ---------
        energy : torch.tensor (shape : (x.shape[0], 1))
            The energy of each x.
        """
        if isinstance(x, dict): # For NUTS sampler
            x = x[0]
        energy, _ = self.calculate_energy(x)
        return energy
