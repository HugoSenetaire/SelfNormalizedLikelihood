from ..EBMsAndMethod import EBMRegression
from ..Energy import get_feature_extractor, get_energy_regression, get_explicit_bias_regression
from ..Proposals import get_proposal_regression, get_base_dist_regression
from .init_proposals import init_energy_to_gaussian_regression, init_proposal_to_data_regression
import numpy as np 



def get_model_regression(args_dict, complete_dataset, complete_masked_dataset, loader_train):
    input_size_x = complete_dataset.get_dim_input()
    input_size_y = complete_dataset.get_dim_output()
    args_dict['input_size_x'] = input_size_x
    args_dict['input_size_y'] = input_size_y

    feature_extractor = get_feature_extractor(input_size_x=input_size_x, args_dict=args_dict)
    if feature_extractor is not None :
        input_size_x_feature = feature_extractor.output_size
    else :
        input_size_x_feature = input_size_x

    # Get energy function :
    energy = get_energy_regression(input_size_x_feature, input_size_y, args_dict)

    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        energy = init_energy_to_gaussian_regression(feature_extractor = feature_extractor, energy = energy, input_size_x = input_size_x, input_size_y = input_size_y, dataloader = loader_train, dataset = complete_dataset.dataset_train, args_dict = args_dict)

    # Get proposal :
    if args_dict['proposal_name'] is not None:
        proposal = get_proposal_regression(args_dict=args_dict, input_size_x_feature=input_size_x_feature, input_size_y=input_size_y, dataset = [complete_dataset.dataset_train, complete_dataset.dataset_val, complete_dataset.dataset_test],)
    else:
        raise ValueError("No proposal given")
    
    if 'proposal_pretraining' in args_dict.keys() and args_dict['proposal_pretraining'] == 'data':
        proposal = init_proposal_to_data_regression(feature_extractor = feature_extractor, proposal = proposal, input_size_x = input_size_x, input_size_y = input_size_y, dataloader = loader_train, args_dict = args_dict)
    # Get base_dist :
    base_dist = get_base_dist_regression(args_dict = args_dict, proposal=proposal, input_size_x_feature=input_size_x_feature, input_size_y=input_size_y, dataset = complete_dataset.dataset_train,)


    explicit_bias = get_explicit_bias_regression(args_dict=args_dict, input_size_x_feature=input_size_x_feature,)


    # Get EBM :
    ebm = EBMRegression(energy = energy, proposal = proposal, feature_extractor= feature_extractor, base_dist = base_dist, explicit_bias = explicit_bias, **args_dict)


    return ebm