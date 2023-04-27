from ..EBMsAndMethod import dic_ebm
from ..Energy import get_energy
from ..Proposals import get_proposal
from ..BaseDist import get_base_dist
from torch.distributions import Normal
import torch
import tqdm
import time

def init_energy_to_gaussian(energy, input_size, dataset, args_dict):
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

def get_model(args_dict, complete_dataset, complete_masked_dataset):
    input_size = complete_dataset.get_dim_input()

    # Get energy function :
    print("Get energy function")
    energy = get_energy(input_size, args_dict)
    print("Get energy function... end")


    if 'ebm_pretraining' in args_dict.keys() and args_dict['ebm_pretraining'] == 'standard_gaussian':
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian(energy, input_size, complete_dataset.dataset_train, args_dict)
        print("Init energy to standard gaussian... end")
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
    print("Get base_dist... end")

    # Get EBM :
    print("Get EBM")
    ebm = dic_ebm[args_dict['ebm_name']](energy = energy, proposal = proposal, base_dist = base_dist, **args_dict)
    print("Get EBM... end")

    return ebm