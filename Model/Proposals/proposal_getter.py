from .ProposalForDistributionEstimation.standard_gaussian import StandardGaussian
from .ProposalForDistributionEstimation.kde import KernelDensity
from .ProposalForDistributionEstimation.kde_adaptive import KernelDensityAdaptive
from .ProposalForDistributionEstimation.gaussian_mixture import GaussianMixtureProposal
from .ProposalForDistributionEstimation.kde import KernelDensity
from .ProposalForDistributionEstimation.poisson import Poisson
from .ProposalForDistributionEstimation.standard_gaussian import StandardGaussian
from .ProposalForRegression.MDNProposal import MDNProposalRegression
from .ProposalForRegression.standard_gaussian import StandardGaussianRegression
from .ProposalForDistributionEstimation.categorical import Categorical
from .ProposalForDistributionEstimation.gaussian_mixture_adaptive import GaussianMixtureAdaptiveProposal
from .ProposalForDistributionEstimation.noise_gradation_adaptive import NoiseGradationAdaptiveProposal
from .ProposalForDistributionEstimation.student import StudentProposal
from .ProposalForRegression import UniformRegression
import torch
import copy

dic_proposals = {
    "standard_gaussian": StandardGaussian,
    "kernel_density": KernelDensity,
    "gaussian_mixture": GaussianMixtureProposal,
    "poisson": Poisson,
    "uniform_categorical": Categorical,
    'kernel_density_adaptive': KernelDensityAdaptive,
    'gaussian_mixture_adaptive': GaussianMixtureAdaptiveProposal,
    'noise_gradation_adaptive' : NoiseGradationAdaptiveProposal,
    'student' : StudentProposal,
}


def get_base_dist(args_dict, proposal, input_size, dataset) :
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)
    base_dist = dic_proposals[args_dict["base_dist_name"]]
    if 'adaptive' in args_dict['base_dist_name'] :
        raise ValueError('Adaptive should only be used for proposal')

    if "base_dist_parameters" not in args_dict:
        args_dict["base_dist_parameters"] = {}
    base_dist = base_dist(input_size, dataset, **args_dict["base_dist_parameters"])
    return base_dist



def get_proposal(args_dict, input_size, dataset, ):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)


    proposal = dic_proposals[args_dict["proposal_name"]]
    if 'proposal_parameters' not in args_dict.keys():
        args_dict['proposal_parameters'] = {}
    if 'adaptive' in args_dict['proposal_name'] :
        assert 'default_proposal_name' in args_dict.keys(), 'You need to specify a default proposal for the adaptive proposal'
        assert not args_dict['train_proposal'], 'You cannot train the proposal if it is adaptive'
        aux_args_dict = copy.deepcopy(args_dict)
        if 'default_proposal_parameters' not in args_dict.keys():
            aux_args_dict['proposal_parameters'] = {}
        else :
            aux_args_dict['proposal_parameters'] = args_dict['default_proposal_parameters']
        aux_args_dict['proposal_name'] = args_dict['default_proposal_name']
        default_proposal = get_proposal(aux_args_dict, input_size, dataset,)
        return proposal(default_proposal = default_proposal, input_size = input_size, dataset = dataset, **args_dict["proposal_parameters"])

    return proposal(input_size, dataset, **args_dict["proposal_parameters"])





dic_proposals_regression = {
    'standard_gaussian': StandardGaussianRegression,
    'mdn': MDNProposalRegression,
    'uniform': UniformRegression,
}



def get_base_dist_regression(args_dict, input_size_x, input_size_y, dataset,):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)
    base_dist = dic_proposals_regression[args_dict["base_dist_name"]]
    if "base_dist_parameters" not in args_dict:
        args_dict['base_dist_parameters'] = {}
    base_dist = base_dist(input_size_x, input_size_y, dataset, **args_dict["base_dist_parameters"])
    return base_dist


def get_proposal_regression(args_dict, input_size_x, input_size_y, dataset,):
    if not isinstance(dataset, list):
        dataset = [dataset]
    dataset = torch.utils.data.ConcatDataset(dataset)
    proposal = dic_proposals_regression[args_dict["proposal_name"]]
    if "proposal_parameters" not in args_dict:
        args_dict['proposal_parameters'] = {}
    proposal = proposal(input_size_x, input_size_y, dataset, **args_dict["proposal_parameters"])
    return proposal
