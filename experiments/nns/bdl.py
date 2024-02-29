import torch
import torch.nn as nn
import push.push as ppush
import torch.nn.functional as F
from torch.utils.data import Dataset
from push.bayes.ensemble import mk_optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# =============================================================================
# Dataset
# =============================================================================

class SineDataset(Dataset):
    def __init__(self, N, D, begin, end):
        self.xs = torch.linspace(begin, end, N).reshape(N, D)
        self.ys = torch.sin(self.xs[:, 0]).reshape(-1, 1) 

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class SineWithNoiseDataset(Dataset):
    def __init__(self, N, D, begin, end, noise_std=0.0005):
        self.xs = torch.linspace(begin, end, N).reshape(N, D)
        true_ys = torch.sin(self.xs[:, 0]).reshape(-1, 1)
        mean = torch.zeros(true_ys.size()[0])
        # Create the tensor with the specified entries
        std = torch.pow(torch.arange(0, N), 1.6) * noise_std
        noise = torch.normal(mean, std).view(-1,1)
        self.ys = true_ys + noise

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class CustomMNISTDataset(Dataset):
    def __init__(self, root, numbers=[0, 1], train=False, transform=None, limit=10):
        self.root = root
        self.train = train
        self.transform = transform
        self.numbers = numbers
        self.limit = limit

        # Download MNIST dataset
        self.mnist_dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=False)
        
        # Filter images based on selected numbers
        indices = np.isin(self.mnist_dataset.targets.numpy(), self.numbers)
        self.data = self.mnist_dataset.data[indices][:self.limit]
        self.targets = self.mnist_dataset.targets[indices][:self.limit]

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.targets[idx])
        
        # Convert to PIL Image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

class SelectMNISTDataset(Dataset):
    def __init__(self, root, numbers=[0, 1], train=False, transform=None, limit=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.numbers = numbers

        # Download MNIST dataset
        self.mnist_dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)

        # Filter images based on selected numbers
        indices = np.isin(self.mnist_dataset.targets.numpy(), self.numbers)
        self.data = self.mnist_dataset.data[indices]
        self.targets = self.mnist_dataset.targets[indices]

        # If limit is specified, truncate the dataset
        if limit is not None:
            self.data = self.data[:limit]
            self.targets = self.targets[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.targets[idx])

        # Convert to PIL Image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# =============================================================================
# Architecture
# =============================================================================

class RegNet(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=1, apply_var=True):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, dimensions, output_dim]        
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())

        if output_dim == 2:
            self.add_module('var_split', SplitDim(correction=apply_var))

class BiggerNN(nn.Module):
    def __init__(self, n, input_dim, output_dim, hidden_dim):
        super(BiggerNN, self).__init__()
        self.minis = []
        self.n = n
       
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        for i in range(0, n):
            self.minis += [MiniNN(hidden_dim)]
            self.add_module("mini_layer"+str(i), self.minis[-1])
        self.fc = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, x):
        x = self.input_layer(x)
        for i in range(0, self.n):
            x = self.minis[i](x)
        return self.fc(x)


class MiniNN(nn.Module):
    def __init__(self, D):
        super(MiniNN, self).__init__()
        self.fc1 = nn.Linear(D, D)
        self.fc2 = nn.Linear(D, D)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        return x

class TwoMoonsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input layer
        self.fc1 = nn.Linear(2, 64)
        self.dropout1 = nn.Dropout(p=0.2)

        # hidden layers
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # output layer
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class GenericNet(nn.Module):
    def __init__(self, input_dim):
        super(GenericNet, self).__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1)
        )
        # Additional layers can be added here based on the architecture
        # Apply Xavier uniform initialization to the weights
        self.init_weights(self.net)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Model(nn.Module):
    def __init__(self, beta):
        super(Model, self).__init__()
        self.prior = GenericNet(input_dim=1)  # Specify the input dimension
        self.trainable = GenericNet(input_dim=1)  # Specify the input dimension
        self.beta = beta

    def forward(self, x):
        x1 = self.prior(x)
        x2 = self.trainable(x)
        return self.beta * x1 + x2



class PriorNet(nn.Module):
    def __init__(self, beta, model, *args):
        super(PriorNet, self).__init__()
        self.prior = model(*args)
        self.trainable = model(*args)
        self.beta = beta

    def forward(self, x):
        x1 = self.prior(x)
        x2 = self.trainable(x)
        return self.beta * x1 + x2
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# =============================================================================
# Model
# =============================================================================

def push_dist_example(num_ensembles, mk_nn, *args):
    # Create a communicating ensemble of particles using mk_nn(*args) as a template.
    with ppush.PusH(mk_nn, *args) as push_dist:
        pids = []
        for i in range(num_ensembles):
            # 1. Each particle has a unique `pid`.
            # 2. `mk_optim` is a particle specific optimization method as in standard PyTorch.
            # 3. Create a particle on device 0.
            # 4. `receive` determines how each particle .
            # 5. `state` is the state associated with each particle.
            pids += [push_dist.p_create(mk_optim, device=0, receive={}, state={})]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    num_ensembles = 4
    input_dim = 1
    output_dim = 1
    hidden_dim = 64
    n = 4
    push_dist_example(num_ensembles, BiggerNN, n, input_dim, output_dim, hidden_dim)
