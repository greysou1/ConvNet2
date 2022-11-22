import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode, image_size, num_classes):
        super(ConvNet, self).__init__()
        
        self.mode = mode
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        # Pool over 2x2 regions, 40 kernels, stride =1, with kernel size of 5x5.
        # define first conv laver 
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        
        # define second conv layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        # define third conv layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        # define forth conv layer
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        # define fifth conv layer
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        # define last conv layer
        self.conv6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        

        # define pool layer
        self.pool = nn.MaxPool2d(2) #kernel_size=(2, 2), stride=(1, 1))
        
        if mode == 1:
            fc_input_size = 4 * 4 * 64
        elif mode == 2:
            fc_input_size = 2 * 2 * 128
        self.batchnorm = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        
        
        self.forward = self.model
        
    # Use two convolutional layers.
    def model(self, X):
        # Three convolutional layers 
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        # X = self.dropout(X)
        X = F.relu(self.conv3(X))
        # Adding more conv layers
        # X = F.relu(self.conv4(X))
        X = self.pool(X)
        # X = self.dropout(X)
        # X = F.relu(self.conv5(X))
        # X = F.relu(self.conv6(X))
        # X = self.pool(X)
        # X = self.dropout(X)
        if self.mode == 2:
            # for _ in range(2):
            X = F.relu(self.conv4(X))
            X = self.dropout(X)
            X = self.pool(X)
            X = F.relu(self.conv5(X))
            X = self.batchnorm(X)
            # X = self.pool(X)
        
        # two fully connected layers, with ReLU. and  + Dropout.
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        # X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return X

    
