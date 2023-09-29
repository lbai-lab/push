import argparse
from datetime import datetime
import pytz
import torch
from timeit import default_timer
import time
from tqdm import tqdm
from typing import *
import sys
sys.path.append('../')
import wandb

from push.bayes.utils import flatten, unflatten_like
from push.bayes.stein_vgd import torch_squared_exp_kernel, torch_squared_exp_kernel_grad
from push.lib.utils import detach_to_cpu, to_device
from train_util import get_model, wandb_init, MyTimer


# =============================================================================
# Multi Swag Training
# =============================================================================

def update_theta(theta, theta2, tt, tt2, n):
    for param in theta.keys():
        theta[param] = (theta[param]*n+tt[param])/(n+1)
        theta2[param] = (theta2[param]*n+tt2[param])/(n+1)

    return theta, theta2


def train_mswag(train_loader, args, loss_fn):
    wandb_init(args, train_loader)

    print("Training Multi-Swag ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    networks = [get_model(args).to(device) for _ in range(args.num_particles)]

    # Create optimizers and schedulers for each model
    optimizers = [torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for net in networks]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=args.scheduler_step, gamma=args.scheduler_gamma) for opt in optimizers]

    # Pretrain epochs
    total_time = 0
    for e in tqdm(range(args.pretrain_epochs), desc="Epochs"):
        losses = []
        with MyTimer() as my_timer:
            for data, label in train_loader:
                preds = []
                for i, model in enumerate(networks):
                    pred = model(to_device(device, data))
                    preds.append(pred)
                    loss = loss_fn(pred, to_device(device, label))

                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
                    losses.append(loss.to("cpu").item())
                
                if args.model == "unet":
                    for scheduler in schedulers:
                        scheduler.step()
        epoch_time = my_timer.time
        total_time += epoch_time
        epoch_loss = torch.mean(torch.tensor(losses))
        if args.wandb:
            wandb.log({
                    "pretrain_epoch_time": epoch_time,
                    "pretrain_loss": epoch_loss,
                })        
    
    mom1s = {i: {name: param for name, param in networks[i].named_parameters()} for i in range(args.num_particles)}
    mom2s = {i: {name: param*param for name, param in networks[i].named_parameters()} for i in range(args.num_particles)}
    optimizers = [torch.optim.Adam(nn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for nn in networks]

    # SWAG epochs
    for e in tqdm(range(args.swag_epochs), desc="Epochs"):
        with MyTimer() as my_timer:
            losses = []
            for data, label in train_loader:
                for idx, (nn, optimizer) in enumerate(zip(networks, optimizers)):
                    pred = nn(to_device(device, data))
                    loss = loss_fn(pred, to_device(device, label))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.to("cpu").item())
                    nn.zero_grad()
                            
            for idx, nn in enumerate(networks):
                params = {name: param for name, param in nn.named_parameters()}
                params_sq = {name: param*param for name, param in nn.named_parameters()}
                update_theta(mom1s[idx], mom2s[idx], params, params_sq, e+1)
        total_time += my_timer.time
        
        if args.wandb:
            wandb.log({
                "swag_epoch_time": my_timer.time,
                "pretrain_loss": torch.mean(torch.tensor(losses)).item(),
            })
    average_time = total_time / args.epochs
    print("Average time per epoch:", average_time)

    return [mom1s, mom2s]
