

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os

class Resnet18_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        resnet18 = models.resnet18(pretrained=True)
        # remove fully connected layer:
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_size = 512

    def forward(self, x):
        # (x has shape (batch_size, 3, img_size, img_size))

        x_feature = self.resnet18(x) # (shape: (batch_size, 512, img_size/32, img_size/32))
        x_feature = self.avg_pool(x_feature) # (shape: (batch_size, 512, 1, 1))
        x_feature = x_feature.squeeze(2).squeeze(2) # (shape: (batch_size, 512))

        return x_feature