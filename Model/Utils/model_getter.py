from ..EBMsAndMethod import dic_ebm, dic_ebm_regression
from ..Energy import get_energy, get_feature_extractor, get_energy_regression, get_explicit_bias_regression
from ..Proposals import get_proposal, get_proposal_regression
from ..BaseDist import get_base_dist, get_base_dist_regression
from torch.distributions import Normal
import torch
import tqdm
import time

def init_energy_to_gaussian(energy, input_size, dataset, args_dict):
    '''
    Initialize the energy to a standard gaussian to make sure it's integrable
    '''
    # dist = Normal(0, 1)
    optimizer = torch.optim.SGD(energy.parameters(), lr=2e-2)

    data = torch.cat([dataset[i][0] for i in range(len(dataset))])
    dist = Normal(data.mean(0), data.std(0))
    ranges= tqdm.tqdm(range(10000))
    for k in ranges:
        x = dist.sample((1000,))
        target_energy = dist.log_prob(x).sum(dim=1)
        # target_energy =  0.5 * (x**2).flatten(1).sum(dim=1)
        current_energy = energy(x).flatten()
        loss = ((current_energy + target_energy)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        ranges.set_description(f'Loss : {loss.item()} Norm of the energy : {current_energy.mean().item()} Norm of the target energy : {target_energy.mean().item()}')
        optimizer.step()
        # time.sleep(1)
    x = dist.sample((10000,))
    log_prob = dist.log_prob(x).sum(dim=1)
    
    norm = (-energy(x).flatten()-log_prob).exp().mean()
    print("=====================================")
    print(f'Norm of the energy : {norm}')
    print("=====================================")
    return energy


def init_energy_to_gaussian_regression(feature_extractor, energy, input_size_x, input_size_y, dataset, args_dict):
    '''
    Initialize the energy to a standard gaussian to make sure it's integrable
    '''
    optimizer = torch.optim.SGD(energy.parameters(), lr=2e-2)
    if feature_extractor is not None :
        data_x_feature = torch.cat([feature_extractor(dataset[i][0]) for i in range(len(dataset))])
    else :
        data_x_feature = torch.cat([dataset[i][0] for i in range(len(dataset))])

    data_y = torch.cat([dataset[i][1] for i in range(len(dataset))])
    dist_y = Normal(data_y.mean(0), data_y.std(0))
    dist_x_feature = Normal(data_x_feature.mean(0), data_x_feature.std(0))
    ranges= tqdm.tqdm(range(10000))
    batch_size = args_dict['batch_size']
    for k in ranges:
        x_feature = dist_x_feature.sample((batch_size,))
        y = dist_y.sample((batch_size, ))
        target_energy = dist_y.log_prob(y).sum(dim=1)
        current_energy = energy(x_feature, y).flatten()
        loss = ((current_energy + target_energy)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        ranges.set_description(f'Loss : {loss.item()} Norm of the energy : {current_energy.mean().item()} Norm of the target energy : {target_energy.mean().item()}')
        optimizer.step()


def get_model(args_dict, complete_dataset, complete_masked_dataset):
    input_size = complete_dataset.get_dim_input()

    # Get energy function :
    print("Get energy function")
    energy = get_energy(input_size, args_dict)
    print("Get energy function... end")


    # Get proposal :
    if args_dict['proposal_name'] is not None:
        print("Get proposal")
        proposal = get_proposal(args_dict=args_dict, input_size=input_size, dataset = complete_dataset.dataset_train,)
        print("Get proposal... end")
    else:
        raise ValueError("No proposal given")
    
    # Get base_dist :
    print("Get base_dist")
    base_dist = get_base_dist(args_dict = args_dict, proposal=proposal, input_size=input_size, dataset = complete_dataset.dataset_train,)
    if args_dict['base_dist_name'] == 'proposal':
        assert proposal == base_dist, "Proposal and base_dist should be the same"
        assert args_dict['train_proposal'] == False, "If training the proposal, the base_dist should not be the proposal"
        for param in proposal.parameters():
            param.requires_grad = False
    print("Get base_dist... end")

    if base_dist is None and ('ebm_pretraining' in args_dict.keys() or args_dict['ebm_pretraining'] is None):
        print("Careful, no base_dist given, the energy might not be well initialized")

    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian(energy, input_size, complete_dataset.dataset_train, args_dict)
        print("Init energy to standard gaussian... end")

    # Get EBM :
    print("Get EBM")
    ebm = dic_ebm[args_dict['ebm_name']](energy = energy, proposal = proposal, base_dist = base_dist, **args_dict)
    print("Get EBM... end")

    return ebm


def get_model_regression(args_dict, complete_dataset, complete_masked_dataset):
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
    print("Get energy function")
    energy = get_energy_regression(input_size_x_feature, input_size_y, args_dict)
    print("Get energy function... end")

    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian_regression(feature_extractor = feature_extractor, energy = energy, input_size_x = input_size_x, input_size_y = input_size_y, dataset = complete_dataset.dataset_train, args_dict = args_dict)
        print("Init energy to standard gaussian... end")

    # Get proposal :
    if args_dict['proposal_name'] is not None:
        print("Get proposal")
        proposal = get_proposal_regression(args_dict=args_dict, input_size_x=input_size_x_feature, input_size_y=input_size_y, dataset = complete_dataset.dataset_train,)
        print("Get proposal... end")
    else:
        raise ValueError("No proposal given")
    
    # Get base_dist :
    print("Get base_dist")
    base_dist = get_base_dist_regression(args_dict = args_dict, proposal=proposal, input_size_x=input_size_x_feature, input_size_y=input_size_y, dataset = complete_dataset.dataset_train,)
    if args_dict['base_dist_name'] == 'proposal':
        assert proposal == base_dist, "Proposal and base_dist should be the same"
    print("Get base_dist... end")
    if args_dict['base_dist_name'] == 'proposal':
        assert args_dict['train_proposal'] == False, "If training the proposal, the base_dist should not be the proposal"
        for param in proposal.parameters():
            param.requires_grad = False

    explicit_bias = get_explicit_bias_regression(args_dict=args_dict, input_size_x=input_size_x_feature,)

    # Get EBM :
    print("Get EBM")
    ebm = dic_ebm_regression[args_dict['ebm_name']](energy = energy, proposal = proposal, feature_extractor= feature_extractor, base_dist = base_dist, explicit_bias = explicit_bias, **args_dict)
    print("Get EBM... end")

    return ebm