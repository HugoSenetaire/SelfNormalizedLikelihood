from .standard_gaussian import StandardGaussian
from .kde import KernelDensity
from .gaussian_mixture import GaussianMixtureProposal


dic_proposals = {
    'standard_gaussian': StandardGaussian,
    'kernel_density': KernelDensity,
    'gaussian_mixture': GaussianMixtureProposal,
}

def get_proposal(args_dict, input_size, dataset):
    proposal = dic_proposals[args_dict['proposal_name']]
    if 'proposal_config' in args_dict:
        return proposal(input_size, dataset, **args_dict['proposal_config'])
    else:
        return proposal(input_size, dataset)