import numpy as np
import torch
from timeit import default_timer
from tqdm import tqdm
from typing import *
import wandb

from push.lib.utils import detach_to_cpu, to_device

import sys
sys.path.append('../')
from train_util import get_model, wandb_init, MyTimer


# =============================================================================
# Deep Ensemble Training
# =============================================================================

def train_deep_ensemble(train_loader,args, loss_fn):
    wandb_init(args, train_loader)
    print("Starting Deep Ensemble Training...")
    print(f'Epochs = {args.epochs}, learning rate = {args.learning_rate}, scheduler step = {args.scheduler_step}, scheduler gamma = {args.scheduler_gamma}')
    
    # Create models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    networks = [get_model(args).to(device) for _ in range(args.num_particles)]

    # Create optimizers and schedulers for each model
    optimizers = [torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for net in networks]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=args.scheduler_step, gamma=args.scheduler_gamma) for opt in optimizers]
    
    total_time = 0
    
    # Main training loop
    tq = tqdm(range(args.epochs))
    for e in tq:
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
        total_time += my_timer.time
        epoch_loss = torch.mean(torch.tensor(losses))
        print("Average Epoch loss", epoch_loss.item())
        
        if args.wandb:
            wandb.log({
                "loss": epoch_loss.item(),
                "time": my_timer.time
            })
            
    average_time = total_time / args.epochs
    print("Average time per epoch:", average_time)
    return networks  # Return the ensemble of models


def test_deep_ensemble(test_loader, args, loss_fn, networks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Starting Deep Ensemble Testing...")
        
    ensemble_losses = []
    ensemble_uncertainties = []
    total_time = 0
    tq = tqdm(test_loader)
    
    for data, label in tq:
        ensemble_pred = []
        t1 = default_timer()
        
        for model in networks:
            model.eval()  # Set the model to evaluation mode
            # with torch.no_grad():
            pred = model(to_device(device, data))
            ensemble_pred.append(pred)
        
        # Average the predictions 
        ensemble_pred = torch.stack(ensemble_pred)
        avg_pred = torch.mean(ensemble_pred, dim=0)
        
        # Measure uncertainty 
        uncertainty = torch.var(ensemble_pred, dim=0)
        ensemble_uncertainties.append(torch.mean(uncertainty).item())

        loss = loss_fn(avg_pred, to_device(device, label))
        ensemble_losses.append(loss.item())
        
        t2 = default_timer()
        total_time += t2 - t1
        
        if args.wandb:
            wandb.log({
                "test_loss": loss.item(),
                "test_uncertainty": torch.mean(uncertainty).item()
            })
        
    avg_test_loss = np.mean(ensemble_losses)
    avg_test_uncertainty = np.mean(ensemble_uncertainties)
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Average Test Uncertainty: {avg_test_uncertainty}")
    print(f"Total Test Time: {total_time}")
    
    return avg_test_loss
