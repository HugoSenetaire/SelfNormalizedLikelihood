import torch
import torch.nn as nn
import numpy as np
import math

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
