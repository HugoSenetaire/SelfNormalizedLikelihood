from ..EBMsAndMethod import ImportanceWeightedEBMDualProposal
from ..Energy import get_energy, get_feature_extractor
from ..Proposals import get_proposal, get_base_dist, get_base_dist_regression
from .init_proposals import init_energy_to_gaussian, init_energy_to_gaussian_regression, init_proposal_to_data, init_proposal_to_data_regression
import numpy as np 




def get_model_distribution_estimation_dual(args_dict, complete_dataset, complete_masked_dataset, loader_train):
    input_size = complete_dataset.get_dim_input()
    args_dict['input_size'] = input_size

    feature_extractor = get_feature_extractor(input_size_x=input_size, args_dict=args_dict)
    if feature_extractor is not None :
        input_size_feature = (1,feature_extractor.output_size)
    else :
        input_size_feature = input_size

    # Get energy function :
    print("Get energy function")
    energy = get_energy(input_size_feature, args_dict)
    print("Get energy function... end")

    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian(feature_extractor = feature_extractor, energy = energy, input_size = input_size, dataloader = loader_train, dataset = complete_dataset.dataset_train, args_dict = args_dict)
        print("Init energy to standard gaussian... end")

    # Get proposal :
    if args_dict['proposal_name'] is not None:
        print("Get proposal")
        proposal = get_proposal(args_dict=args_dict, input_size=input_size_feature, dataset = [complete_dataset.dataset_train, complete_dataset.dataset_val, complete_dataset.dataset_test], feature_extractor=feature_extractor)
        print("Get proposal... end")
    else:
        raise ValueError("No proposal given")
    
    if 'proposal_pretraining' in args_dict.keys() and args_dict['proposal_pretraining'] == 'data':
        print("Init proposal")
        proposal = init_proposal_to_data(feature_extractor = feature_extractor, proposal = proposal, input_size = input_size, dataloader = loader_train, args_dict = args_dict)
        print("Init proposal... end")
    
    # Get base_dist :
    print("Get base_dist")
    base_dist = get_base_dist(args_dict = args_dict, proposal=proposal, input_size=input_size_feature, dataset = complete_dataset.dataset_train,)
    if args_dict['base_dist_name'] == 'proposal':
        assert proposal == base_dist, "Proposal and base_dist should be the same"
    elif args_dict['base_dist_name'] == 'none' :
        base_dist = None
    else :
        for param in base_dist.parameters():
            param.requires_grad = True
    print("Get base_dist... end")
    if args_dict['base_dist_name'] == 'proposal':
        assert args_dict['train_proposal'] == False, "If training the proposal, the base_dist should not be the proposal"
        for param in proposal.parameters():
            param.requires_grad = False


    # Get EBM :
    print("Get EBM")
    ebm = ImportanceWeightedEBMDualProposal(energy = energy, proposal = proposal, feature_extractor = feature_extractor, base_dist = base_dist, **args_dict)
    print("Get EBM... end")

    return ebm