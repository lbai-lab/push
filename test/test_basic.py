import argparse
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import push.bayes.ensemble
import push.bayes.stein_vgd
import push.bayes.swag

import time


torch.manual_seed(34)


"""[Note]

Quick Usage

1. `python test_basic.py -m ensemble`
2. `python test_basic.py -m stein_vgd`
3. `python test_basic.py -m mswag`

"""

# =============================================================================
# Simple Dataset + Neural Network
# =============================================================================

class RandDataset(Dataset):
    def __init__(self, batch_size, N, D):
        self.xs = torch.randn(batch_size*N, D)
        self.ys = torch.randn(batch_size*N, 1)

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


# =============================================================================
# Basic Testing
# =============================================================================

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--layers", type=int, default=2,
                        help="size of neural network")
    parser.add_argument("-D", "--dimension", type=int, default=20,
                        help="size of neural network")
    parser.add_argument("-p", "--particles", type=int, default=2,
                        help="number of particles for SVGD or mswag")
    parser.add_argument("-b", "--batchsize", type=int, default=128,
                        help="number of batches")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of batches")
    parser.add_argument("-n", "--size", type=int, default=10,
                        help="size of dataset")
    parser.add_argument("-d", "--devices", type=int, default=1,
                        help="number of devices")
    parser.add_argument("-cs", "--cache_size", type=int, default=4,
                        help="size of the cache")
    parser.add_argument("-vs", "--view_size", type=int, default=4,
                        help="size of the cache")
    parser.add_argument("-m", "--method", type=str, default="ensemble",
                        choices=[
                            "ensemble",
                            "swag-central",
                            "swag",
                            "mswag",
                            "stein_vgd",
                        ],
                        help="bayesian inference method")
    args = parser.parse_args()

    # Dataset
    L = args.layers
    D = args.dimension
    N = args.size
    dataset = RandDataset(args.batchsize, N, D)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    
    # Dispatch
    if args.method == "ensemble":
        def mk_dataloader():
            return DataLoader(RandDataset(args.batchsize, args.size, args.dimension), batch_size=128, shuffle=True)

        epochs = 10
        num_ensembles = 3
        ensemble = push.bayes.ensemble.train_deep_ensemble(
            dataloader, # mk_dataloader,
            torch.nn.MSELoss(),
            epochs,
            BiggerNN, L, D,
            num_devices=args.devices,
            num_ensembles=num_ensembles
        )
        print("mean", ensemble.posterior_pred(dataloader, mode="mean"))
        print("min", ensemble.posterior_pred(dataloader, mode="min"))
        print("median", ensemble.posterior_pred(dataloader, mode="median"))
        print("max", ensemble.posterior_pred(dataloader, mode="max"))

    elif args.method == "mswag":
        pretrain_epochs = 10
        swag_epochs = 5
        cache_size=4
        view_size=4
        start = time.perf_counter()
        push.bayes.swag.train_mswag(
            dataloader,
            torch.nn.MSELoss(),
            pretrain_epochs,
            swag_epochs,
            args.particles,
            cache_size,
            view_size, 
            BiggerNN, L, D
        )
        print("Time elapsed", time.perf_counter() - start)

    elif args.method == "stein_vgd":
        import numpy as np
        start = time.perf_counter()
        print("Size of model", sum([np.prod(p.size()) for p in BiggerNN(L, D).parameters()]))
        print("Size of dataset", len(dataloader.dataset))
        svgd_state = {
            "L": L,
            "D": D,
            "N": N,
        }
        ensemble = push.bayes.stein_vgd.train_svgd(
            dataloader, torch.nn.MSELoss(),
            args.epochs, args.particles,
            BiggerNN, L, D,
            lengthscale=1.0, lr=1e-3, prior=None,
            cache_size=args.cache_size, view_size=args.view_size, num_devices=args.devices,
            svgd_entry=push.bayes.stein_vgd._svgd_leader, svgd_state=svgd_state
        )
        print("Time elapsed", time.perf_counter() - start)
        print("mean", ensemble.posterior_pred(dataloader, mode="mean"))
        print("min", ensemble.posterior_pred(dataloader, mode="min"))
        print("median", ensemble.posterior_pred(dataloader, mode="median"))
        print("max", ensemble.posterior_pred(dataloader, mode="max"))

    else:
        raise ValueError(f"Method {args.method} not supported ...")
