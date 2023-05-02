from .ProposalForDistributionEstimation.standard_gaussian import StandardGaussian
from .ProposalForDistributionEstimation.kde import KernelDensity
from .ProposalForDistributionEstimation.gaussian_mixture import GaussianMixtureProposal
from .ProposalForRegression.standard_gaussian import StandardGaussianRegression
from .ProposalForRegression.MDNProposal import MDNProposalRegression

dic_proposals = {
    'standard_gaussian': StandardGaussian,
    'kernel_density': KernelDensity,
    'gaussian_mixture': GaussianMixtureProposal,
}


def get_proposal(args_dict, input_size, dataset):
    proposal = dic_proposals[args_dict['proposal_name']]
    if 'proposal_params' in args_dict:
        return proposal(input_size, dataset, **args_dict['proposal_params'])
    else:
        return proposal(input_size, dataset)
    

dic_proposals_regression = {
    'standard_gaussian': StandardGaussianRegression,
    'mdn': MDNProposalRegression,
}
    
def get_proposal_regression(args_dict, input_size_x, input_size_y, dataset):
    proposal = dic_proposals_regression[args_dict['proposal_name']]
    if 'proposal_params' in args_dict:
        return proposal(input_size_x, input_size_y, dataset, **args_dict['proposal_params'])
    else:
        return proposal(input_size_x, input_size_y, dataset)