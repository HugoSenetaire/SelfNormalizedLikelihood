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

import copy

dic_proposals = {
    "standard_gaussian": StandardGaussian,
    "kernel_density": KernelDensity,
    "gaussian_mixture": GaussianMixtureProposal,
    "poisson": Poisson,
    "uniform_categorical": Categorical,
    'kernel_density_adaptive': KernelDensityAdaptive,
    'gaussian_mixture_adaptive': GaussianMixtureAdaptiveProposal,
}



def get_proposal(args_dict, input_size, dataset):

    proposal = dic_proposals[args_dict["proposal_name"]]
    if 'adaptive' in args_dict['proposal_name'] :
        assert 'default_proposal_name' in args_dict.keys(), 'You need to specify a default proposal for the adaptive proposal'
        assert not args_dict['train_proposal'], 'You cannot train the proposal if it is adaptive'
        if isinstance(dataset, list):
            current_dataset = dataset[0]
        if 'proposal_params' not in args_dict.keys():
            args_dict['proposal_params'] = {}
        aux_args_dict = copy.deepcopy(args_dict)
        if 'default_proposal_params' not in args_dict.keys():
            aux_args_dict['proposal_params'] = {}
        else :
            aux_args_dict['proposal_params'] = args_dict['default_proposal_params']
        
        aux_args_dict['proposal_name'] = args_dict['default_proposal_name']
        default_proposal = get_proposal(aux_args_dict, input_size, current_dataset,)
        return proposal(default_proposal = default_proposal, input_size = input_size, dataset = current_dataset, **args_dict["proposal_params"])

    if "proposal_params" in args_dict:
        return proposal(input_size, current_dataset, **args_dict["proposal_params"])
    else:
        return proposal(input_size, dataset)

from .ProposalForRegression import UniformRegression

dic_proposals_regression = {
    'standard_gaussian': StandardGaussianRegression,
    'mdn': MDNProposalRegression,
    'uniform': UniformRegression,
}


def get_proposal_regression(args_dict, input_size_x, input_size_y, dataset,):
    proposal = dic_proposals_regression[args_dict["proposal_name"]]
    if "proposal_params" in args_dict:
        proposal = proposal(input_size_x, input_size_y, dataset, **args_dict["proposal_params"])
        return proposal
    else:
        return proposal(input_size_x, input_size_y, dataset)
