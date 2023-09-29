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

from push.lib.utils import detach_to_cpu, to_device
from train_util import get_model


# =============================================================================
# Standard Training
# =============================================================================

def train_standard(train_loader, args, loss_fn):
    print("Starting Standard Training...")
    print(f'Epochs = {args.epochs}, learning rate = {args.learning_rate}, scheduler step = {args.scheduler_step}, scheduler gamma = {args.scheduler_gamma}')
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)

    # Create optimizers and schedulers for each particle
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    total_time = 0
    tq = tqdm(range(args.epochs))
    for e in tq:
        losses = []
        t1 = default_timer()
        for data, label in train_loader:
            pred = model(to_device(device, data))
            loss = loss_fn(pred, to_device(device, label))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.to("cpu")]
        
        if args.model == "unet":
            scheduler.step()
        
        t2 = default_timer()
        total_time += t2 - t1
        epoch_loss = torch.mean(torch.tensor(losses))
        print("Average Epoch loss", epoch_loss)

        if args.wandb:
            wandb.log({
                    "loss": torch.mean(torch.tensor(losses)).item(),
                    "time": t2 - t1})
            
    average_time = total_time / args.epochs
    print("Average time per epoch:", average_time)
    return model


def test_standard(test_loader, args, loss_fn, trained_model):
    print("Testing Standard Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_losses = []
    total_batches = 0
    for data, label in tqdm(test_loader, desc="testloader"):
        pred = trained_model(to_device(device, data))
        test_loss = loss_fn(pred, to_device(device, label))
        test_losses.append(detach_to_cpu(test_loss).item())
        total_batches += 1

        if args.wandb:
            wandb.log({"test loss " + str(args.dataset): test_loss.item()})
    # Compute average test loss
    average_test_loss = sum(test_losses) / total_batches
    print(f"Average Test Loss: {average_test_loss}")
