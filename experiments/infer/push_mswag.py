from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from typing import *
import wandb

import push.bayes.swag as swag
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu


import sys
sys.path.append('../')
from train_util import wandb_init, MyTimer


# =============================================================================
# PusH Multi-Swag Training
# =============================================================================

def _mswag_particle_instrumented(particle: Particle,
                                 dataloader,
                                 loss_fn: Callable,
                                 pretrain_epochs: int,
                                 swag_epochs: int, swag_pids: list[int]) -> None:
    state = particle.state
    wandb_init(state["args"], dataloader)
    other_pids = [pid for pid in swag_pids if pid != particle.pid]
    
    # Pre-training loop
    for e in tqdm(range(pretrain_epochs)):
        losses = []
        with MyTimer() as my_timer:
            for data, label in dataloader:
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "SWAG_STEP", loss_fn, data, label) for pid in other_pids]
                losses += [fut.wait()]
                [f.wait() for f in futs]
        if state["args"].wandb:
            wandb.log({
                "pretrain_epoch_time": my_timer.time,
                "pretrain_loss:": torch.mean(torch.tensor(losses))
            })

    # Initialize swag
    with MyTimer() as my_timer:
        futs = [particle.send(pid, "SWAG_SWAG", True) for pid in other_pids]
        swag._swag_swag(particle, True)
        [f.wait() for f in futs]
    if state["args"].wandb:
        wandb.log({
            "send_time": my_timer.time
        })

    # Swag epochs
    for e in tqdm(range(swag_epochs)):
        losses = []
        with MyTimer() as my_timer:
            for data, label in dataloader:
                # Update
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "SWAG_STEP", loss_fn, data, label) for pid in other_pids]
                [f.wait() for f in futs]
                losses += [fut.wait()]
            swag._swag_swag(particle, False)
            futs = [particle.send(pid, "SWAG_SWAG", False) for pid in other_pids]
            [f.wait() for f in futs]
        if state["args"].wandb:
            wandb.log({
                "swag_epoch_time": my_timer.time,
                "leader_loss:": torch.mean(torch.tensor(losses))
            })


# =============================================================================
# PusH Multi-Swag Inference
# =============================================================================

def _mswag_sample_instrumented2(particle: Particle,
                  dataloader: DataLoader,
                  loss_fn: Callable,
                  scale: float,
                  var_clamp: float,
                  num_samples: int,
                  num_models) -> None:
    state = particle.state
    pid = particle.pid

    other_pids = list(range(1, num_models))
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [particle.send(pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples) for pid in other_pids]
    ans = [f.wait() for f in futs]
    
    classes = {k: [0 for i in range(10)] for k in range(10)}
    max_preds = [my_ans['max_preds']] + [ans[i-1]['max_preds'] for i in range(1, num_models)]
    for idx, (data, label) in enumerate(dataloader):
        max_pred = torch.mode(torch.stack([max_preds[i][idx] for i in range(num_models)]), dim=0).values
        for x, y in zip(max_pred, label):
            classes[y.item()][x.item()] += 1
    
    all_preds = [my_ans['preds']] + [ans[i-1]['preds'] for i in range(1, num_models)]
    classes2 = {k: [0 for i in range(10)] for k in range(10)}
    for idx, (data, label) in enumerate(dataloader):
        preds = []
        for m in range(num_models):
            preds += [torch.stack([all_preds[m][i][idx] for i in range(num_samples)])]
        max_pred = torch.mode(torch.cat(preds, dim=0), dim=0).values
        for x, y in zip(max_pred, label):
            classes2[y.item()][x.item()] += 1

    if state["args"].wandb:
        wandb.log({
            f"max_dist{pid}": str(classes),
            f"flat_max_dist{pid}": str(classes2),
        })


def _mswag_sample(particle: Particle,
                  dataloader: DataLoader,
                  loss_fn: Callable,
                  scale: float,
                  var_clamp: float,
                  num_samples: int) -> None:
    """Inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    """
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    classes = {k: [0 for i in range(10)] for k in range(10)}
    losses = []
    for data, label in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, label)
        cls = pred.softmax(dim=1).argmax(dim=1)
        for x, y in zip(cls, label):
            classes[y.item()][x.item()] += 1
        losses += [loss.detach().to("cpu")]

    if state["args"].wandb:
        wandb.log({
            f"orig_loss{pid}": torch.mean(torch.tensor(losses)),
            f"orig_dist{pid}": str(classes)
        })

    preds = {i: [] for i in range(num_samples)}
    swag_losses = {}
    for i in range(num_samples):
        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        swag_losses = []
        for data, label in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            cls = pred.softmax(dim=1).argmax(dim=1)
            preds[i] += [cls]
            loss = loss_fn(pred, label)
            swag_losses += [loss.detach().to("cpu")]    

        if state["args"].wandb:
            wandb.log({
                f"swag_loss{pid}": torch.mean(torch.tensor(swag_losses)),
                f"classes_dist{pid}": str(classes)
            })
    
    max_preds = []
    for n in range(len(preds[0])):
        max_preds += [torch.mode(torch.stack([preds[i][n] for i in range(num_samples)]), dim=0).values]

    return {
        "classes": classes, 
        "losses": torch.mean(torch.tensor(losses)),
        "preds": preds,
        "max_preds": max_preds,
    }


def _mswag_sample_instrumented(particle: Particle,
                  dataloader: DataLoader,
                  loss_fn: Callable,
                  scale: float,
                  var_clamp: float,
                  num_samples: int) -> None:
    """Inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    """
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    classes = {k: [0 for i in range(10)] for k in range(10)}
    losses = []
    for data, label in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, label)
        cls = pred.softmax(dim=1).argmax(dim=1)
        for x, y in zip(cls, label):
            classes[y.item()][x.item()] += 1
        losses += [loss.detach().to("cpu")]
    if state["args"].wandb:
        wandb.log({
            f"orig_loss{pid}": torch.mean(torch.tensor(losses)),
            f"orig_dist{pid}": str(classes)
        })

    for _ in range(num_samples):
        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        # with torch.no_grad():
        #     for param, sample in zip(particle.module.parameters(), samples_list):
        #         param.copy_(sample)
        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        classes = {k: [0 for i in range(10)] for k in range(10)}
        swag_losses = []
        for data, label in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            cls = pred.softmax(dim=1).argmax(dim=1)
            for x, y in zip(cls, label):
                classes[y.item()][x.item()] += 1
            loss = loss_fn(pred, label)
            swag_losses += [loss.detach().to("cpu")]

        if state["args"].wandb:
            wandb.log({
                f"swag_loss{pid}": torch.mean(torch.tensor(swag_losses)),
                f"classes_dist{pid}": str(classes)
            })