import unittest
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import push.bayes.ensemble
import push.bayes.stein_vgd
import push.bayes.swag


# =============================================================================
# Simple Dataset + Neural Network
# =============================================================================

class RandDataset(Dataset):
    def __init__(self, D):
        self.xs = torch.randn(128*10, D)
        self.ys = torch.randn(128*10, 1)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


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
    

class BiggerNN(nn.Module):
    def __init__(self, n, D):
        super(BiggerNN, self).__init__()
        self.minis = []
        self.n = n
        for i in range(0, n):
            self.minis += [MiniNN(D)]
            self.add_module("mini_layer"+str(i), self.minis[-1])
        self.fc = nn.Linear(D, 1)
            
    def forward(self, x):
        for i in range(0, self.n):
            x = self.minis[i](x)
        return self.fc(x)


class TestEnd2End(unittest.TestCase):
    def test_ensemble(self):
        L = 10
        D = 20
        dataset = RandDataset(D)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        epochs = 10
        num_ensembles = 3
        push.bayes.ensemble.train_deep_ensemble(
            dataloader,
            torch.nn.MSELoss(),
            epochs,
            BiggerNN, L, D,
            num_ensembles=num_ensembles
        )

    def test_swag(self):
        L = 10
        D = 20
        dataset = RandDataset(D)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        pretrain_epochs = 10
        swag_epochs = 5
        cache_size = 4
        view_size = 4
        num_models = 2
        push.bayes.swag.train_mswag(
            dataloader,
            torch.nn.MSELoss(),
            pretrain_epochs,
            swag_epochs,
            num_models,
            cache_size,
            view_size, 
            BiggerNN, L, D, num_devices=1
        )

    def test_stein_vgd(self):
        L = 10
        D = 20
        epochs = 3
        particles = 4
        devices = 1
        cache_size = 4
        view_size = 4
        dataset = RandDataset(D)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        svgd_state = {
            "L": L,
            "D": D,
            "N": 10,
        }
        push.bayes.stein_vgd.train_svgd(
            dataloader, torch.nn.MSELoss(),
            epochs, particles,
            BiggerNN, L, D,
            lengthscale=1.0, lr=1e-3, prior=None,
            cache_size=cache_size, view_size=view_size, num_devices=devices,
            svgd_entry=push.bayes.stein_vgd._svgd_leader, svgd_state=svgd_state
        )
