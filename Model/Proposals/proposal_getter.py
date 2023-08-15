import copy

import torch

from .mock_base_dist import MockBaseDist
from .ProposalForDistributionEstimation.categorical import get_Categorical
from .ProposalForDistributionEstimation.gaussian import get_Gaussian
from .ProposalForDistributionEstimation.gaussian_mixture import (
    get_GaussianMixtureProposal,
)
from .ProposalForDistributionEstimation.gaussian_mixture_adaptive import (
    get_GaussianMixtureAdaptiveProposal,
)
from .ProposalForDistributionEstimation.kde import get_KernelDensity
from .ProposalForDistributionEstimation.maf import get_MAFProposal
from .ProposalForDistributionEstimation.noise_gradation_adaptive import (
    get_NoiseGradationAdaptiveProposal,
)
from .ProposalForDistributionEstimation.gaussian_full import get_GaussianFull
from .ProposalForDistributionEstimation.poisson import get_Poisson
from .ProposalForDistributionEstimation.pytorch_flows import get_PytorchFlowsProposal
from .ProposalForDistributionEstimation.real_nvp_proposal import get_RealNVPProposal
from .ProposalForDistributionEstimation.student import get_StudentProposal
from .ProposalForDistributionEstimation.vera_proposal import get_vera, get_vera_hmc
from .ProposalForRegression import get_UniformRegression
from .ProposalForRegression.gaussian import get_GaussianRegression
from .ProposalForRegression.MDNProposal import get_MDNProposalRegression
from .ProposalForRegression.uniform import get_UniformRegression


dic_proposals = {
    "gaussian": get_Gaussian,
    "kernel_density": get_KernelDensity,
    "gaussian_mixture": get_GaussianMixtureProposal,
    "poisson": get_Poisson,
    "uniform_categorical": get_Categorical,
    "gaussian_mixture_adaptive": get_GaussianMixtureAdaptiveProposal,
    "noise_gradation_adaptive": get_NoiseGradationAdaptiveProposal,
    "student": get_StudentProposal,
    "real_nvp": get_RealNVPProposal,
    "vera_hmc": get_vera_hmc,
    "vera": get_vera,
    "maf": get_MAFProposal,
    "pytorch_flows": get_PytorchFlowsProposal,
    'gaussian_full': get_GaussianFull,
}


def get_base_dist(cfg, proposal, input_size, dataset):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)
    if cfg.base_distribution is None or cfg.base_distribution.proposal_name is None:
        return MockBaseDist()
    if cfg.base_distribution.proposal_name == "proposal":
        return proposal
    base_dist = dic_proposals[cfg.base_distribution.proposal_name]
    if "adaptive" in cfg.base_distribution.proposal_name:
        raise ValueError("Adaptive should only be used for proposal")

    base_dist = base_dist(input_size, dataset, cfg.base_distribution)
    return base_dist


def get_proposal(
    cfg,
    input_size,
    dataset,
):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)

    proposal = dic_proposals[cfg.proposal.proposal_name]
    if "adaptive" in cfg.proposal.proposal_name:
        assert (
            cfg.default_proposal is not None
        ), "You need to specify a default proposal for the adaptive proposal"
        assert (
            not cfg.proposal_training.train_proposal
        ), "You cannot train the proposal if it is adaptive"
        # aux_args_dict = copy.deepcopy(args_dict)
        default_proposal = dic_proposals[cfg.default_proposal.proposal_name](
            input_size, dataset, cfg.default_proposal
        )
        return proposal(
            default_proposal=default_proposal,
            input_size=input_size,
            dataset=dataset,
            cfg=cfg.proposal,
        )

    return proposal(input_size, dataset, cfg.proposal)


dic_proposals_regression = {
    "gaussian": get_GaussianRegression,
    "mdn": get_MDNProposalRegression,
    "uniform": get_UniformRegression,
}


def get_base_dist_regression(
    cfg,
    proposal,
    input_size_x_feature,
    input_size_y,
    dataset,
):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)

    # If there is no base distribution, we return None
    if cfg.base_distribution is None or cfg.base_distribution.proposal_name is None:
        return MockBaseDist()
    # If the base distribution is the proposal, we return the proposal
    if cfg.base_distribution.proposal_name == "proposal":
        assert (
            "adaptive" not in cfg.base_distribution.proposal_name
        ), "Adaptive proposal should not be used as base distribution"
        return proposal

    base_dist = dic_proposals_regression[cfg.base_distribution.proposal_name]

    base_dist = base_dist(
        input_size_x_feature, input_size_y, dataset, cfg.base_distribution
    )

    return base_dist


def get_proposal_regression(
    cfg,
    input_size_x_feature,
    input_size_y,
    dataset,
):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)

    proposal = dic_proposals_regression[cfg.proposal.proposal_name]
    proposal = proposal(input_size_x_feature, input_size_y, dataset, cfg.proposal)
    return proposal
