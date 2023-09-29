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
# SVGD Training
# =============================================================================

def svgd_step_precompute( n, networks, bandwidth, lr): #parameterwise version
    network_params = [[] for _ in range(n)]

    for i in range(n):
        network_params[i] = [param for param in networks[i].parameters()]

    K = {i: torch.zeros(n, device="cuda") for i in range(n)}
    K_grad = {i: [] for i in range(n)}
    
    for i in range(n):
        params1 = list(network_params[i]) 
        p_i = flatten(params1)
        for j in range(n):
            update = torch.zeros_like(p_i)
            params2 = list(network_params[j]) 
            # Compute kernel
            p_j = flatten(params2)
            diff = (p_j - p_i) / bandwidth
            radius2 = torch.dot(diff, diff)
            k = torch.exp(-0.5 * radius2).item()
            diff.mul_(-k / bandwidth)
            update.add_(diff.real)
            K[i][j] = k
            K_grad[i] += [update]
            # K[i][j] = torch_squared_exp_kernel(p_i, p_j, bandwidth).real
            # K_grad[i] += [torch_squared_exp_kernel_grad(p_i, p_j, bandwidth).real]    
   
    for i in range(n):
        acc = [torch.zeros_like(p) for p in network_params[i]]
        for j in range(n):

            pj = list(network_params[j])
            k_grads_j = unflatten_like(K_grad[j][i].unsqueeze(0),pj) 
            
            for idx, tmp in enumerate(acc): 
                #print(K[i][j].dtype, k_grads_j[idx].dtype, pj[idx].grad.dtype, tmp.dtype)
                if network_params[j][idx].grad is not None:
                    tmp.add_(pj[idx].grad, alpha=K[j][i].item()/n)
                tmp.add_(k_grads_j[idx].real, alpha=1/n) 
        with torch.no_grad():
            for idx, p in enumerate(network_params[i]):
                if p.grad is not None:
                    p.add_(acc[idx], alpha=-lr)


def train_svgd(train_loader, args, loss_fn):
    wandb_init(args, train_loader)

    print("Training SVGD.....")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    networks = [get_model(args).to(device) for _ in range(args.num_particles)]
    total_time = 0
    for e in tqdm(range(args.epochs), desc="Epochs"):
        losses = []
        with MyTimer() as my_timer:
            for data, label in train_loader:
                # Gradients
                for network in networks:
                    network.zero_grad()
                    pred = network(to_device(device, data))
                    loss = loss_fn(pred, to_device(device, label))
                    losses += [loss.to("cpu").item()]
                    loss.backward()  

                svgd_step_precompute( args.num_particles, networks, args.bandwidth, args.learning_rate)
        epoch_time = my_timer.time
        total_time += epoch_time
        print("Time per svgd epoch:", epoch_time)
        print("Average loss:", torch.mean(torch.tensor(losses)))
        if args.wandb:
            wandb.log({
                "epoch_loss": torch.mean(torch.tensor(losses)).item(),
                "time": epoch_time,
            })
    if args.wandb:
        wandb.log({
            "average time": total_time/args.epochs,         
        })
    return networks


def test_svgd(test_loader, args, loss_fn, networks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing test_loss")
    test_losses = []
    
    for data, label in test_loader:
        preds = []
        
        #Average predictions 
        for network in networks:
            preds.append(network(to_device(device, data)))
        preds_stacked = torch.stack(preds, dim=0)
        pred_mean = torch.mean(preds_stacked, dim=0)

        test_loss = loss_fn(pred_mean, to_device(device, label))
        if args.wandb:
            wandb.log({"test loss " + args.dataset: test_loss})
        test_losses += [test_loss.to("cpu")]
    avg_test_loss = torch.mean(torch.tensor(test_losses))
    print("Average Test Loss: Standard", avg_test_loss)
    if args.wandb:
        wandb.log({"average test loss " + args.dataset: avg_test_loss})
