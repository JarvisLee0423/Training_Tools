'''
    @Copyright:     JarvisLee
    @Date:          2021/1/31
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the model.
class Model(nn.Module):
    # Create the constructor.
    def __init__(self):
        super(Model, self).__init__()
        # Set the first conv layer.
        self.conv1 = nn.Conv2d(1, 64, (3, 3), padding = 1)
        # Set the first max pooling layer.
        self.maxPool1 = nn.MaxPool2d((2, 2), 2)
        # Set the second conv layer.
        self.conv2 = nn.Conv2d(64, 128, (5, 5))
        # Set the second max pooling layer.
        self.maxPool2 = nn.MaxPool2d((2, 2), 2)
        # Set the last conv layer.
        self.lastconv = nn.Conv2d(128, 64, (1, 1))
        # Set the first linear layer.
        self.linear1 = nn.Linear(1600, 400)
        # Set the last linear layer.
        self.linear2 = nn.Linear(400, 10)
    # Define the forward propagation.
    def forward(self, x):
        # Apply the first conv layer.
        x = self.conv1(x)
        # Apply the first max pooling layer.
        x = self.maxPool1(x)
        # Apply the second conv layer.
        x = self.conv2(x)
        # Apply the second max pooling layer.
        x = self.maxPool2(x)
        # Apply the last conv layer.
        x = self.lastconv(x)
        # Flatten the data.
        x = x.view(-1, 1600)
        # Apply the first linear layer.
        x = self.linear1(x)
        # Apply the second linear layer.
        x = self.linear2(x)
        # Return the prediction.
        return x