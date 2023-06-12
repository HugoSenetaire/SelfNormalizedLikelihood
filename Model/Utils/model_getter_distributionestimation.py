from ..EBMsAndMethod import EBMRegression, ImportanceWeightedEBM
from ..Energy import get_energy, get_feature_extractor, get_energy_regression, get_explicit_bias_regression
from ..Proposals import get_proposal, get_proposal_regression, get_base_dist, get_base_dist_regression
from .init_proposals import init_energy_to_gaussian, init_energy_to_gaussian_regression, init_proposal_to_data, init_proposal_to_data_regression
import numpy as np 


def get_model(args_dict, complete_dataset, complete_masked_dataset, loader_train):
    input_size = complete_dataset.get_dim_input()

    # Get energy function :
    print("Get energy function")
    energy = get_energy(input_size, args_dict)
    print("Get energy function... end")


    # Get proposal :
    if args_dict['proposal_name'] is not None:
        print("Get proposal")
        proposal = get_proposal(args_dict=args_dict, input_size=input_size, dataset = [complete_dataset.dataset_train, complete_dataset.dataset_val, complete_dataset.dataset_test],)
        print("Get proposal... end")
    else:
        raise ValueError("No proposal given")
    
    # Get base_dist :
    print("Get base_dist")
    base_dist = get_base_dist(args_dict = args_dict, proposal=proposal, input_size=input_size, dataset = complete_dataset.dataset_train,)
    if args_dict['base_dist_name'] == 'proposal':
        assert proposal == base_dist, "Proposal and base_dist should be the same"
        # assert args_dict['train_proposal'] == False, "If training the proposal, the base_dist should not be the proposal"
        for param in proposal.parameters():
            param.requires_grad = False
    print("Get base_dist... end")

    if base_dist is None and ('ebm_pretraining' not in args_dict.keys() or args_dict['ebm_pretraining'] is None):
        print("Careful, no base_dist given, the energy might not be well initialized")

    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian(energy, input_size, complete_dataset.dataset_train, args_dict)
        print("Init energy to standard gaussian... end")

    if 'proposal_pretraining' in args_dict.keys() and args_dict['proposal_pretraining'] == 'data':
        print("Init proposal")
        proposal = init_proposal_to_data(proposal = proposal, input_size = input_size, dataloader = loader_train, args_dict = args_dict)
        print("Init proposal... end")

    # Get EBM :
    print("Get EBM")
    ebm = ImportanceWeightedEBM(energy = energy, proposal = proposal, base_dist = base_dist, **args_dict)
    print("Get EBM... end")

    return ebm
