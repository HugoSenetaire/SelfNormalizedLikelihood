import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torchvision


def plot_energy_2d(algo, save_dir, energy_function=None, name = 'contour_best', samples = [], samples_title = [], step='', energy_type = True):
    """
    Plot the energy of the EBM in 2D and the data points
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if energy_function is None:
        energy_function = lambda x: algo.ebm.calculate_energy(x)[0]
    nx = 1000
    ny = 1000
    min_x, max_x = algo.min_x, algo.max_x
    min_y, max_y = algo.min_y, algo.max_y

    for s, title in zip(samples, samples_title) :
        min_x, max_x = min(torch.min(s[:,0],), min_x), max(torch.max(s[:,0]), max_x)
        min_y, max_y = min(torch.min(s[:,1],), min_y), max(torch.max(s[:,1]), max_y)

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)

    
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    xy = torch.from_numpy(xy).float()
    if energy_type :
        z = (-energy_function(xy)).exp().detach().cpu().numpy()
    else :
        z = energy_function(xy).detach().cpu().numpy()
    z = z.reshape(nx, ny)
    assert len(samples_title) == len(samples)
    fig, axs = plt.subplots(1,  2 + len(samples), figsize=(10 + 5 *  len(samples),5))

    axs[0].contourf(x,y,z, 100)
    axs[0].set_title('Energy')
    for i, (s,s_title) in enumerate(zip(samples,samples_title)) :
        axs[i+1].contourf(x, y, z, 100)
        axs[i+1].scatter(s[:,0], s[:,1], c='r', alpha = 0.1)
        axs[i+1].set_title(s_title)
    fig.colorbar(axs[0].contourf(x,y,z, 100), cax=axs[-1])
    plt.savefig(os.path.join(save_dir, "{}_{}.png".format(name, step)))
    try :
        algo.logger.log_image(key = "{}.png".format(name,), images = [fig])
    except AttributeError as e :
        print(e, )
    plt.close()

       
def plot_energy_1d_regression(algo, save_dir, energy_function = None, name = 'contour_best', samples_x = [], samples_y = [], samples_title = [], step='', energy_type = True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if energy_function is None:
        energy_function = lambda x,y: algo.ebm.calculate_energy(x,y)[0] 

    nx = 1000
    ny = 1000
    min_x, max_x = algo.min_x, algo.max_x
    min_y, max_y = algo.min_y, algo.max_y

    for s_x, s_y, title in zip(samples_x, samples_y, samples_title) :
        min_x, max_x = min(torch.min(s_x,), min_x), max(torch.max(s_x), max_x)
        min_y, max_y = min(torch.min(s_y,), min_y), max(torch.max(s_y), max_y)

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)

    
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    xy = torch.from_numpy(xy).float()
    if energy_type :
        z = (-energy_function(xy[:,0,None,], xy[:,1, None])).exp().detach().cpu().numpy()
    else :
        z = energy_function(xy[:,0,None,], xy[:,1, None]).detach().cpu().numpy()
    z = z.reshape(nx, ny)
    assert len(samples_title) == len(samples_x)
    assert len(samples_title) == len(samples_y)
    fig, axs = plt.subplots(1,  2 + len(samples_x), figsize=(10 + 5 *  len(samples_x),5))

    axs[0].contourf(x,y,z, 100)
    axs[0].set_title('Energy')
    for i, (s_x, s_y, s_title) in enumerate(zip(samples_x, samples_y, samples_title)) :
        axs[i+1].contourf(x, y, z, 100)
        axs[i+1].scatter(s_x, s_y, c='r', alpha = 0.1)
        axs[i+1].set_title(s_title)
    fig.colorbar(axs[0].contourf(x,y,z, 100), cax=axs[-1])
    plt.savefig(os.path.join(save_dir, "{}_{}.png".format(name, step)))
    try :
        algo.logger.log_image(key = "{}.png".format(name,), images = [fig])
    except AttributeError as e :
        print(e, )
    print("Saved at ", os.path.join(save_dir, "{}_{}.png".format(name, step)))
    plt.close()


# def sample_from_energy(energy_function, input_size, proposal = None, num_samples = None):
#     torch.set_grad_enabled(True)
#     samples = nuts_sampler(energy_function, input_size = input_size, proposal=proposal, num_samples=num_samples, )
#     samples = samples[0]
#     return samples

def plot_images(images, save_dir, transform_back = None, algo = None, name = 'samples', init_samples= None, step='', ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if transform_back is not None :
        images = transform_back(images)
        if len(images.shape) == 3 :
            images = images.unsqueeze(1)

    if init_samples is not None :
        init_samples = transform_back(init_samples)
        if len(init_samples.shape) == 3 :
            init_samples = init_samples.unsqueeze(1)
        images = torch.cat([init_samples, images], dim = 0)
    grid = torchvision.utils.make_grid(images,)
    torchvision.utils.save_image(grid, os.path.join(save_dir, "{}_{}.png".format(name, step)))
    try :
        algo.logger.log_image(key = "{}.png".format(name,), images = [grid])
    except AttributeError as e :
        print(e, )
