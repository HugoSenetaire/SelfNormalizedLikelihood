import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.parametrizations import spectral_norm

def get_ConvEnergy(input_size, cfg, ):
    return ConvEnergy(input_size,
                        cfg.activation,
                        cfg.last_layer_bias,
                        )

def get_ConvEnergy_nijkamp(input_size, cfg, ):
    return conv_nijkamp(
                        cfg.nijkamp_n_c,
                        cfg.nijkamp_n_f,
                        cfg.nijkamp_l,
                        sn=False
                        )

def get_ConvEnergy_nijkamp_sn(input_size, cfg, ):
    return conv_nijkamp(
                        cfg.nijkamp_n_c,
                        cfg.nijkamp_n_f,
                        cfg.nijkamp_l,
                        sn=True
                        )

class ConvEnergy(nn.Module):
    def __init__(self, input_size = (1,28,28), activation = None, last_layer_bias = False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.activation = activation
        self.nb_block = int(math.log(min(self.input_size[1], self.input_size[2]), 2)//2)
        
        liste_conv = []
        liste_conv.extend([
            nn.Conv2d(input_size[0], 2**5, 3, stride=1, padding=1),
            nn.Conv2d(2**5, 2**5, 3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        ])
        for k in range(1, self.nb_block):
            liste_conv.extend([
                nn.Conv2d(2**(k+4), 2**(k+5), 3, stride=1, padding=1),
                nn.Conv2d(2**(k+5), 2**(k+5), 3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0),
            ]
            )
        self.conv = nn.ModuleList(liste_conv)
        last_channel = 2**(self.nb_block+4)
        last_size = int(np.prod(input_size[1:])/(2**(2*self.nb_block)))
        self.fc = nn.Linear(last_channel*last_size,128)

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(128,1, bias=last_layer_bias)


    
    def __call__(self, x):
        x = x.view(-1, *self.input_size)
        for k in range(len(self.conv)):
            x = self.conv[k](x)
        x = x.view(x.shape[0],-1)
        x = self.elu(self.fc(x))
        x = self.fc2(x)
        if self.activation is not None :
            x = self.activation(x)
        return x.reshape(-1,1)


class conv_nijkamp(nn.Module):
    def __init__(self, n_c = 1, n_f = 64, l = 0.2, sn = False):
        super(conv_nijkamp, self).__init__()
        if not sn :
            self.f = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                nn.LeakyReLU(l),
                nn.Conv2d(n_f, n_f*2, 4, 2, 1),
                nn.LeakyReLU(l),
                nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
                nn.LeakyReLU(l),
                nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
                nn.LeakyReLU(l),
                nn.Conv2d(n_f*8, 1, 4, 1, 0),
            )
        else :
            self.f = nn.Sequential(
                spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                nn.LeakyReLU(l),
                spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                nn.LeakyReLU(l),
                spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                nn.LeakyReLU(l),
                spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                nn.LeakyReLU(l),
                spectral_norm(nn.Conv2d(n_f * 8, 1, 4, 1, 0)))


    def forward(self, x):
        return self.f(x).squeeze()
    

