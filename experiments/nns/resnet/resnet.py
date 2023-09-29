import torch.nn as nn
import torchvision


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(in_features=self.resnet18.fc.in_features, out_features=10)

    def forward(self, x):
        return self.resnet18(x)
