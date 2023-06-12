# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os




class EnergyNetworkRegression_Large(nn.Module):

    def __init__(self, input_size_x_feature, input_size_y):
        super().__init__()
        self.input_size_y = input_size_y
        self.input_dim_y = np.prod(self.input_size_y)
        self.input_size_x_feature = input_size_x_feature
        self.input_dim_x_feature = np.prod(self.input_size_x_feature)


        self.fc1_y = nn.Linear(self.input_dim_y, 16)
        self.fc2_y = nn.Linear(16, 32)
        self.fc3_y = nn.Linear(32, 64)
        self.fc4_y = nn.Linear(64, 128)

        self.fc1_xy = nn.Linear(self.input_dim_x_feature+128, self.input_dim_x_feature)
        self.fc2_xy = nn.Linear(self.input_dim_x_feature, 1)

    def forward(self, x_feature, y):
        x_feature = x_feature.reshape(-1, self.input_dim_x_feature)
        y = y.reshape(-1, self.input_dim_y)
        assert x_feature.shape[0] == y.shape[0]
        assert x_feature.shape[1] == self.input_dim_x_feature
        try :
            assert y.shape[1] == self.input_dim_y
        except AssertionError:
            y = y.unsqueeze(1)
            assert y.shape[1] == self.input_dim_y
        except IndexError:
            y = y.unsqueeze(1)
            assert y.shape[1] == self.input_dim_y


        # replicate:
        y_feature = F.relu(self.fc1_y(y)) # (shape: (batch_size*num_samples, 16))
        y_feature = F.relu(self.fc2_y(y_feature)) # (shape: (batch_size*num_samples, 32))
        y_feature = F.relu(self.fc3_y(y_feature)) # (shape: (batch_size*num_samples, 64))
        y_feature = F.relu(self.fc4_y(y_feature)) # (shape: (batch_size*num_samples, 128))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, hidden_dim+128))

        xy_feature = F.relu(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc2_xy(xy_feature) # (shape: (batch_size*num_samples, 1))
        # score = score.view(batch_size, num_samples) # (shape: (batch_size, num_samples))

        return score.reshape(-1,1) # (shape: (batch_size, num_samples))




class EnergyNetworkRegression_Toy(nn.Module):
    def __init__(self, input_size_x_feature=10, input_size_y=1, hidden_dim = 50):
        super().__init__()
        self.input_size_y = input_size_y
        self.input_dim_y = np.prod(self.input_size_y)
        self.input_size_x_feature = input_size_x_feature
        self.input_dim_x_feature = np.prod(self.input_size_x_feature)
        hidden_dim = max(self.input_dim_x_feature, hidden_dim)

        
        self.fc1_y = nn.Linear(self.input_dim_y, hidden_dim)
        self.fc2_y = nn.Linear(hidden_dim, hidden_dim)

        self.fc1_xy = nn.Linear(hidden_dim + self.input_dim_x_feature, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_xy = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))
        x_feature = x_feature.reshape(-1, self.input_dim_x_feature) 
        y = y.reshape(-1, self.input_dim_y) # (shape: (batch_size*num_samples, 1))
        assert x_feature.shape[0] == y.shape[0]
        assert x_feature.shape[1] == self.input_dim_x_feature
        assert y.shape[1] == self.input_dim_y


        y_feature = F.relu(self.fc1_y(y)) # (shape: (batch_size*num_samples, hidden_dim))
        y_feature = F.relu(self.fc2_y(y_feature)) # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, 2*hidden_dim))
        xy_feature = F.relu(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = F.relu(self.fc2_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc3_xy(xy_feature) # (shape: (batch_size*num_samples, 1))
        return score.reshape(-1,1) # (shape: (batch_size, num_samples))