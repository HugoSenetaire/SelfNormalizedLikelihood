import torch
import torch.distributions as distributions
import itertools
import torch.nn as nn


class BiasExplicit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.explicit_bias = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=True)
    
    def forward(self, x):
        return x + self.explicit_bias

class ImportanceWeightedEBMDualProposal(nn.Module):
    def __init__(
        self,
        energy,
        proposal_total,
        proposal_feature,
        num_sample_proposal,
        feature_extractor,
        base_dist=None,
        bias_explicit=False,
        nb_sample_bias_explicit=1024,
        **kwargs,
    ):
        super(ImportanceWeightedEBMDualProposal, self).__init__()
        self.energy = energy
        self.proposal_total = proposal_total
        self.proposal_feature = proposal_feature
        self.nb_sample = num_sample_proposal
        self.nb_sample_bias_explicit = nb_sample_bias_explicit
        self.base_dist = base_dist
        self.bias_explicit = bias_explicit
        self.feature_extractor = feature_extractor

        if bias_explicit:
            # self.explicit_bias = torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=True)
            self.explicit_bias_module = BiasExplicit() 
            self.explicit_bias = self.explicit_bias_module.explicit_bias
            log_z_estimate, dic = self.estimate_log_z(torch.zeros(1, dtype=torch.float32,), nb_sample = self.nb_sample_bias_explicit)
            self.explicit_bias_module.explicit_bias.data = log_z_estimate
        else :
            self.explicit_bias_module = None

    def get_features(self, x):
        x_feature = self.feature_extractor(x)
        return x_feature

    def sample_total(self, nb_sample=1):
        """
        Samples from the proposal distribution.
        """
        return self.proposal_total.sample(nb_sample)

    def sample_feature(self, nb_sample=1):
        """
        Samples from the proposal feature distribution.
        """
        return self.proposal_feature.sample(nb_sample)

    def calculate_energy(self, x, use_base_dist=True, x_feature = None,):
        """
        Calculate energy of x with the energy function
        """
        dic_output = {}
        if x_feature is None :
            x_feature = self.get_features(x)
        out_energy = self.energy(x_feature)
        dic_output["f_theta"] = out_energy

        if (self.training and self.proposal_total is not None and hasattr(self.proposal,"set_x",)):
            self.proposal_total.set_x(x)
        if (self.training and self.proposal_feature is not None and hasattr(self.proposal_feature,"set_x",)):
            self.proposal_feature.set_x(x_feature)

        if self.base_dist is not None and use_base_dist:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            base_dist_log_prob = (
                self.base_dist.log_prob(x).view(x.size(0), -1).sum(1).unsqueeze(1)
            )
            dic_output["base_dist_log_prob"] = base_dist_log_prob
        else:
            base_dist_log_prob = torch.zeros_like(out_energy)

        if self.explicit_bias_module is not None :
            out_energy = self.explicit_bias_module(out_energy)
            dic_output.update({"log_bias_explicit" : self.explicit_bias_module.explicit_bias})


        current_energy = out_energy - base_dist_log_prob
        dic_output["energy"] = current_energy

        return current_energy, dic_output



    def switch_mode(
        self,
    ):
        """
        Switch the mode of the model and perform renormalization when moving from one mode to another.
        """
        if self.bias_explicit:
            samples = self.sample(self, nb_sample=self.nb_sample_bias_explicit)
            log_prob_samples = self.proposal.log_prob(samples)
            energy_samples = self.calculate_energy(samples)
            estimated_z = (-energy_samples - log_prob_samples).exp().mean()
            self.log_Z.data = torch.log(estimated_z)



    def estimate_log_z(self, x, nb_sample=1000, x_feature = None):
        dic_output = {}
        if x_feature is None :
            x_feature = self.get_features(x)
        
        samples = self.sample(nb_sample).to(x.device, x.dtype)
        energy_samples = self.energy(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
        if self.bias_explicit :
            energy_samples = self.explicit_bias_module(energy_samples)
        dic_output['f_theta_samples'] = energy_samples
        
        if self.base_dist is not None :
            base_dist_log_prob = self.base_dist.log_prob(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
            if self.base_dist != self.proposal_feature :
                base_dist_log_prob = self.base_dist.log_prob(samples).view(samples.size(0), -1).sum(1).unsqueeze(1)
                samples_log_prob = self.proposal_total.log_prob(samples).reshape(samples.shape[0], -1).sum(1).unsqueeze(1)
                aux_prob = base_dist_log_prob - samples_log_prob 
                log_z_estimate = torch.logsumexp(-energy_samples + aux_prob,dim=0) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device)) 
                z_estimate = log_z_estimate.exp()
                dic_output.update({
                        "base_dist_log_prob_samples": base_dist_log_prob,
                        "proposal_log_prob_samples": samples_log_prob,
                        "aux_prob_samples": aux_prob,}
                        )
            else:
                log_z_estimate = (-energy_samples.flatten()).logsumexp(
                    dim=0
                ) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
                z_estimate = log_z_estimate.exp()
        else:
            samples_log_prob = (
                self.proposal.log_prob(samples)
                .view(samples.size(0), -1)
                .sum(1)
                .unsqueeze(1)
            )
            dic_output.update(
                {
                    "proposal_log_prob_samples": samples_log_prob,
                }
            )
            log_z_estimate = torch.logsumexp(
                -energy_samples - samples_log_prob, dim=0
            ) - torch.log(torch.tensor(nb_sample, dtype=x_feature.dtype, device=x_feature.device))
            z_estimate = log_z_estimate.exp()

        dic_output["log_z_estimate"] = log_z_estimate
        dic_output["z_estimate"] = z_estimate
        return log_z_estimate, dic_output

    def forward(self, x):
        """
        Hade to create a separate forward function in order to pickle the model and multiprocess HMM.
        """
        current_x = x[0].to(
            next(self.energy.parameters()).device, next(self.energy.parameters()).dtype
        )
        # current_x_feature = self.get_features(current_x)
        energy, _ = self.calculate_energy(x)
        return energy
