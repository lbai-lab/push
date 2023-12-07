import torch
import torch.nn as nn
from torch.utils.data import Dataset

import push.push as ppush
from push.bayes.ensemble import mk_optim


# =============================================================================
# Dataset
# =============================================================================

class SineDataset(Dataset):
    def __init__(self, batch_size, N, D, begin, end):
        self.xs = torch.linspace(begin, end, batch_size * N * D).reshape(batch_size * N, D)
        self.ys = torch.sin(self.xs[:, 0]).reshape(-1, 1) 

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class SineWithNoiseDataset(Dataset):
    def __init__(self, N, D, begin, end, noise_std=0.05):
        self.xs = torch.linspace(begin, end, N).reshape(N, D)
        true_ys = torch.sin(self.xs[:, 0]).reshape(-1, 1)
        noise = torch.normal(0, noise_std, size=true_ys.size())
        self.ys = true_ys + noise

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

# class SineWithNoiseDataset(Dataset):
#     def __init__(self, batch_size, N, D, begin, end, noise_std=0.05):
#         self.xs = torch.linspace(begin, end, batch_size * N * D).reshape(batch_size * N, D)
#         true_ys = torch.sin(self.xs[:, 0]).reshape(-1, 1)
#         noise = torch.normal(0, noise_std, size=true_ys.size())
#         self.ys = true_ys + noise

#     def __len__(self):
#         return len(self.xs)

#     def __getitem__(self, idx):
#         return self.xs[idx], self.ys[idx]



# =============================================================================
# Architecture
# =============================================================================

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
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Model
# =============================================================================

def push_dist_example(num_ensembles, mk_nn, *args, cache_size=8, view_size=8):
    # Create a communicating ensemble of particles using mk_nn as a template
    with ppush.PusH(mk_nn, *args, cache_size=cache_size, view_size=view_size) as push_dist:
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
