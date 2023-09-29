import argparse
from datetime import datetime
import pytz
import torch
from timeit import default_timer
import time
from tqdm import tqdm
import wandb
from typing import *
from torch.utils.data import DataLoader

import push.bayes.stein_vgd as svgd
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu

import sys
sys.path.append('../')
from train_util import wandb_init, MyTimer


# =============================================================================
# PusH Stein VGD Training
# =============================================================================

def _svgd_leader_instrumented(particle: Particle,
                              prior, loss_fn: Callable, lengthscale, lr,
                              dataloader: DataLoader, epochs) -> None:
    state = particle.state
    wandb_init(state["args"], dataloader)

    n = len(particle.particle_ids())
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

    for e in tqdm(range(epochs)):
        losses = []
        epoch_time = 0
        for data, label in dataloader:
            with MyTimer() as my_timer:
                # 1. Step every particle       
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "SVGD_STEP", loss_fn, data, label) for pid in other_particles]
                fut.wait(); [f.wait() for f in futs]

                # 2. Gather every other particles parameters
                particles = {pid: (particle.get(pid) if pid != particle.pid else 
                    list(particle.module.parameters())) for pid in particle.particle_ids()}
                for pid in other_particles:
                    particles[pid] = particles[pid].wait()

                # 3. Compute kernel and kernel gradients
                update = {}
                for pid1, params1 in particles.items():
                    params1 = list(particles[pid1].view().parameters()) if pid1 != particle.pid else params1
                    p_i = flatten(params1)
                    update[pid1] = torch.zeros_like(p_i)
                    for pid2, params2 in particles.items():
                        params2 = list(particles[pid2].view().parameters()) if pid2 != particle.pid else params2

                        # Compute kernel
                        p_j = flatten(params2)
                        p_j_grad = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in params2])
                        update[pid1] += svgd.torch_squared_exp_kernel(p_j, p_i, lengthscale) * p_j_grad
                        update[pid1] += svgd.torch_squared_exp_kernel_grad(p_j, p_i, lengthscale).real
                    update[pid1] = update[pid1] / n

                # 4. Send kernel
                futs = [particle.send(pid, "SVGD_FOLLOW", lr, update[pid]) for pid in other_particles]
                [f.wait() for f in futs]
                svgd._svgd_follow(particle, lr, update[particle.pid])
            epoch_time += my_timer.time


            losses += [loss_fn(detach_to_cpu(particle.forward(data).wait()), label)]
        if state["args"].wandb:
            wandb.log({
                "time": epoch_time,
                "leader_loss:": torch.mean(torch.tensor(losses))
            })


def _svgd_leader_instrumented_memeff(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs) -> None:
    print("DOING 2")
    state = particle.state
    wandb_init(state["args"], dataloader)

    n = len(particle.particle_ids())
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

    for e in tqdm(range(epochs)):
        losses = []
        epoch_time = 0
        for data, label in tqdm(dataloader,desc="data_loader"):
            with MyTimer() as my_timer:
                
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "SVGD_STEP", loss_fn, data, label) for pid in other_particles]
                fut.wait(); [f.wait() for f in futs]

                # 2. Gather every other particles parameters
                particles = {pid: (particle.get(pid) if pid != particle.pid else 
                    list(particle.module.parameters())) for pid in particle.particle_ids()}
                for pid in other_particles:
                    particles[pid] = particles[pid].wait()

                def compute_update(pid1, params1):
                    params1 = list(particles[pid1].view().parameters()) if pid1 != particle.pid else params1
                    p_i = flatten(params1)
                    update = torch.zeros_like(p_i)
                    for pid2, params2 in particles.items():
                        params2 = list(particles[pid2].view().parameters()) if pid2 != particle.pid else params2

                        # Compute kernel
                        p_j = flatten(params2)
                        p_j_grad = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in params2])
                        diff = (p_j - p_i) / lengthscale
                        radius2 = torch.dot(diff, diff)
                        k = torch.exp(-0.5 * radius2).item()
                        diff.mul_(-k / lengthscale)
                        update.add_(p_j_grad, alpha=k.real)
                        update.add_(diff.real)
                        # update += svgd.torch_squared_exp_kernel(p_j, p_i, lengthscale).real * p_j_grad
                        # update += svgd.torch_squared_exp_kernel_grad(p_j, p_i, lengthscale).real
                    update = update / n
                    return update

                # 3. Compute kernel and kernel gradients
                for pid1, params1 in particles.items():
                    # 4. Send kernel
                    if pid1 != particle.pid:
                        update = compute_update(pid1, params1)
                        particle.send(pid, "SVGD_FOLLOW", lr, update).wait()
                update = compute_update(particle.pid, particles[particle.pid])
                svgd._svgd_follow(particle, lr, update)
            epoch_time += my_timer.time
            # Record losses
            loss= loss_fn(detach_to_cpu(particle.forward(data).wait()), label) 
            losses += [loss]
        if state["args"].wandb:
            wandb.log({
                "time": epoch_time,
                "leader_loss:": torch.mean(torch.tensor(losses))
            })
