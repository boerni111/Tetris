import torch
from torch import nn

class DQN_model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1)

        self.fc_1 = nn.Linear(216, 32)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(32, n_actions)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        return x