import torch.distributions as dist
import numpy as np
import torch
import torch.nn as nn

class StandardGaussian(nn.Module):
    def __init__(self, input_size, dataset, mean='dataset', std ='dataset', **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        print("Init Standard Gaussian...")
        
        # try :
        if isinstance(dataset,list):
            current_dataset = dataset[0]
        else :
            current_dataset = dataset
        index = np.random.choice(len(current_dataset), 10000)
        
        # print(current_dataset.__getitem__(0)[0].shape)
        data = torch.cat([current_dataset.__getitem__(i)[0] for i in index]).reshape(-1, *self.input_size)
        # data = torch.cat([current_dataset[i][0] for i in index]).reshape(-1, *self.input_size)
        # except TypeError :
            # data = torch.cat([dataset[i][0][0] for i in index]).reshape(-1, *self.input_size)
        if mean == 'dataset' :
            self.mean = nn.parameter.Parameter(data.mean(0), requires_grad=False)
        else :
            raise NotImplementedError
        import matplotlib.pyplot as plt

        if std == 'dataset' :
            self.std = nn.parameter.Parameter(data.std(0), requires_grad=False)
        else :
            raise NotImplementedError
        
        # print(torch.sigmoid(self.mean))
        # plt.imshow(torch.sigmoid(self.mean[0]).detach().cpu().numpy())
        # plt.savefig("mean.png")
        # plt.imshow(torch.sigmoid(self.std[0]).detach().cpu().numpy())
        # plt.savefig("std.png")

 

        
        self.distribution = dist.Normal(self.mean, self.std)
        # test = self.distribution.sample(64)
        # from torchvision.utils import save_image, make_grid
        # save_image(make_grid(test), "test.png")
        # assert False
        print("Init Standard Gaussian... end")

    def sample(self, nb_sample = 1):
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, *self.input_size).detach()
        return samples
    
    def log_prob(self, x):
        return self.distribution.log_prob(x).flatten(1).sum(1)
    
