
from torch.distributions import Normal
import torch
import tqdm
import numpy as np
import time


def init_energy_to_gaussian(energy, input_size, dataset, args_dict):
    '''
    Initialize the energy to a standard gaussian to make sure it's integrable
    '''
    # dist = Normal(0, 1)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    energy = energy.to(device)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    data = torch.cat([dataset[i][0] for i in range(len(dataset))]).reshape(-1,*dataset[0][0].shape).flatten(1).to(device)
    dist = Normal(data.mean(0), data.std(0))
    ranges= tqdm.tqdm(range(10000))
    for k in ranges:
        x = dist.sample((1000,)).to(device)
        target_energy = -dist.log_prob(x).reshape(1000,-1).sum(dim=1).to(device)
        # target_energy =  0.5 * (x**2).flatten(1).sum(dim=1)
        current_energy = energy(x).reshape(1000,).to(device)
        loss = ((current_energy - target_energy)**2).mean()
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
    energy = energy.to(torch.device('cpu'))

    return energy


def init_energy_to_gaussian_regression(feature_extractor, energy, input_size_x, input_size_y, dataloader, dataset, args_dict):
    '''
    Initialize the energy to a standard gaussian to make sure it's integrable for all the inputs
    '''
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    energy = energy.to(device)
    if feature_extractor is not None :
        feature_extractor = feature_extractor.to(device)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)
    
    data_y = torch.cat([dataset[i][1].unsqueeze(0) for i in range(len(dataset))]).to(device, dtype)
    dist_y = Normal(data_y.mean(0), data_y.std(0))
    epochs = 20
    batch_size = args_dict['batch_size']
    for k in range(epochs):
        ranges = tqdm.tqdm(enumerate(dataloader))
        for batch_idx, batch in ranges:
            x = batch['data'].to(device, dtype)
            if feature_extractor is not None :
                x_feature = feature_extractor(x)
            else :
                x_feature = x
            y = dist_y.sample((x.shape[0], )).to(device, dtype).reshape(x.shape[0], -1)

            target_energy = -dist_y.log_prob(y).reshape(x.shape[0], -1).to(device, dtype)
            current_energy = energy(x_feature, y).reshape(x.shape[0], -1)
            loss = ((current_energy - target_energy)**2).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            if batch_idx % 10 == 0 :
                ranges.set_description(f'Loss : {loss.item()} Norm energy : {current_energy.mean().item()} Norm target energy : {target_energy.mean().item()}')
            optimizer.step()
    energy = energy.to(torch.device('cpu'))
    if feature_extractor is not None :
        feature_extractor = feature_extractor.to(torch.device('cpu'))
    return energy



def init_proposal_to_data(proposal, input_size, dataloader, args_dict):
    '''
    Initialize the proposal to the data
    '''
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    proposal = proposal.to(device)

    for param in proposal.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(proposal.parameters(), lr=1e-3)
    print("Init proposal to data")
    for epoch in range(100):
        tqdm_range = tqdm.tqdm(dataloader)
        for batch in tqdm_range:
            x = batch['data']
            x = x.to(device, dtype)
            log_prob = proposal.log_prob(x,).reshape(-1)
            loss = (-log_prob).mean()
            tqdm_range.set_description(f'Loss : {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    proposal = proposal.to(torch.device('cpu'))
    print("Init proposal to data... end")
    return proposal

def init_proposal_to_data_regression(feature_extractor, proposal, input_size_x, input_size_y, dataloader, args_dict):
    '''
    Initialize the proposal to the data
    '''
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    proposal = proposal.to(device)
    if feature_extractor is not None :
        feature_extractor = feature_extractor.to(device)
    for param in proposal.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(proposal.parameters(), lr=1e-4)
    print("Init proposal to data")
    for epoch in range(10):
        tqdm_range = tqdm.tqdm(dataloader)
        for batch in tqdm_range:
            x, y = batch['data'], batch['target']
            x = x.to(device, dtype)
            y = y.to(device, dtype)
            if feature_extractor is not None :
                x = feature_extractor(x)
            log_prob = proposal.log_prob(x, y).reshape(-1)
            loss = (-log_prob).mean()
            tqdm_range.set_description(f'Loss : {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    proposal = proposal.to(torch.device('cpu'))
    if feature_extractor is not None :
        feature_extractor = feature_extractor.to(torch.device('cpu'))
    print("Init proposal to data... end")
    return proposal
