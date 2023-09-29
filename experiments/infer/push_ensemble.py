import torch
from tqdm import tqdm
import wandb
from typing import *

from push.particle import Particle

import sys
sys.path.append('../')
from train_util import wandb_init, MyTimer


# =============================================================================
# Push Ensemble Training
# =============================================================================

def mk_optim(lr, weight_decay, params):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _ensemble_main_instrumented(particle: Particle, dataloader, loss_fn, epochs) -> None:
    state = particle.state
    wandb_init(state["args"], dataloader)

    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
    # Training loop
    for e in tqdm(range(epochs)):
        losses = []
        with MyTimer() as my_timer:
            for data, label in dataloader:
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "ENSEMBLE_STEP", loss_fn, data, label) for pid in other_particles]
                losses += [fut.wait()]
                [f.wait() for f in futs]

        # print(f"Average loss {particle.pid}", torch.mean(torch.tensor(losses)))
        if state["args"].wandb:
            wandb.log({
                "time": my_timer.time,
                "leader_loss:": torch.mean(torch.tensor(losses))
            })
