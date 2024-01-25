import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    Inspired by mnist_classification_mc_dropout.ipynb <https://github.com/probml/pyprobml/blob/master/notebooks/book2/17/mnist_classification_mc_dropout.ipynb> from "Probabilistic Machine learning" by Kevin Murphy 
    
    "Gradient-based learning applied to document recognition" <https://ieeexplore.ieee.org/document/726791> 
    """
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding="same"),   # 28*28->32*32-->28*28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),  # 27*27
            
            # #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding="same"),  # 10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),  # 26*26

            #3
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1), # 25*25
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120*25*25, out_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=84, out_features=10)
            
        )
        
    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)
