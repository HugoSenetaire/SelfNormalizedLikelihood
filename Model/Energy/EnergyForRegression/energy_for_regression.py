# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os




class EnergyNetworkRegression_Large(nn.Module):

    def __init__(self, input_dim_x, input_dim_y):
        super().__init__()
        self.input_dim_y = np.prod(input_dim_y)
        self.input_dim_x = np.prod(input_dim_x)
        self.fc1_y = nn.Linear(self.input_dim_y, 16)
        self.fc2_y = nn.Linear(16, 32)
        self.fc3_y = nn.Linear(32, 64)
        self.fc4_y = nn.Linear(64, 128)

        self.fc1_xy = nn.Linear(self.input_dim_x+128, self.input_dim_x)
        self.fc2_xy = nn.Linear(self.input_dim_x, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, input_dim_y)) 
        assert x_feature.shape[0] == y.shape[0]
        assert x_feature.shape[1] == self.input_dim_x
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

        return score




class EnergyNetworkRegression_Toy(nn.Module):
    def __init__(self, input_dim_x=10, input_dim_y=1, hidden_dim = 50):
        super().__init__()
        self.input_dim_y = np.prod(input_dim_y)
        self.input_dim_x = np.prod(input_dim_x)
        hidden_dim = max(self.input_dim_x, hidden_dim)
        self.fc1_y = nn.Linear(self.input_dim_y, hidden_dim)
        self.fc2_y = nn.Linear(hidden_dim, hidden_dim)

        self.fc1_xy = nn.Linear(hidden_dim + self.input_dim_x, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_xy = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        assert x_feature.shape[0] == y.shape[0]
        assert x_feature.shape[1] == self.input_dim_x
        assert y.shape[1] == self.input_dim_y



        # resize to batch dimension
        x_feature = x_feature.reshape(-1, self.input_dim_x) # (shape: (batch_size*num_samples, hidden_dim))
        y = y.reshape(-1, self.input_dim_y) # (shape: (batch_size*num_samples, 1))

        # print(x_feature.mean())
        # print(y.mean())
        y_feature = F.relu(self.fc1_y(y)) # (shape: (batch_size*num_samples, hidden_dim))
        y_feature = F.relu(self.fc2_y(y_feature)) # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, 2*hidden_dim))
        xy_feature = F.relu(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = F.relu(self.fc2_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc3_xy(xy_feature) # (shape: (batch_size*num_samples, 1))
        return score