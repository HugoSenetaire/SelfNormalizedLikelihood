

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import os



class ConvFeatureExtractorDistributionMnist(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.output_size = (1,512)
        

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.max_pool_3(x)
        x = x.view(x.size(0), *self.output_size)
        return x