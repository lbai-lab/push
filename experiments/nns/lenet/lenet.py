import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 1st layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1)
        # 2nd layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2, stride=1)
        # 3rd layer
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=2, stride=1)  # Adjusted kernel size and stride

        # Fully connected layers
        # Adjust the input size of fc1 to match the output size of conv3
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 1st layer
        x = F.relu(F.avg_pool2d(self.conv1(x), 2))

        # 2nd layer
        x = F.relu(F.avg_pool2d(self.conv2(x), 2))

        # 3rd layer
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten

        # 4th layer
        x = F.relu(self.fc1(x))

        # 5th layer
        x = self.fc2(x)

        return x
