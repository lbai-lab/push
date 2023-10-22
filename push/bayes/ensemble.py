
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *

from push.bayes.infer import Infer
from push.particle import Particle


# =============================================================================
# Helper
# =============================================================================

def mk_optim(params):
    """returns Adam optimizer"""
    # Limitiation must be global
    return torch.optim.Adam(params, lr=1e-1, weight_decay=1e-2)


# =============================================================================
# Deep Ensemble
# =============================================================================

def _deep_ensemble_main(particle: Particle, dataloader: DataLoader, loss_fn: Callable, epochs: int) -> None:
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
    # Training loop
    for e in tqdm(range(epochs)):
        losses = []
        for data, label in dataloader:
            loss = particle.step(loss_fn, data, label).wait()
            losses += [loss]
            for pid in other_particles:
                particle.send(pid, "ENSEMBLE_STEP", loss_fn, data, label)
        print(f"Average loss {particle.pid}", torch.mean(torch.tensor(losses)))


def _ensemble_step(particle: Particle, loss_fn: Callable, data, label, *args) -> None:
    particle.step(loss_fn, data, label, *args)


class Ensemble(Infer):
    """
    The base class for any Ensemble Bayesian Deep Learning task
    
    :param Callable mk_nn: The base model to be ensembled.
    :param any *args: Any arguments required for base model to be initialized.
    :param int num_devices: The desired number of gpu devices that will be utilized.
    :param int cache_size: The size of cache used to store particles.
    :param int view_size: The number of particles to consider storing in cache.
    """
    def __init__(self, mk_nn: Callable, *args: any, num_devices: int = 1, cache_size: int = 4, view_size: int = 4) -> None:
        super(Ensemble, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        
    def bayes_infer(self,
                    dataloader: DataLoader, epochs: int,
                    loss_fn=torch.nn.MSELoss(),
                    num_ensembles: int = 2, mk_optim=mk_optim,
                    ensemble_entry=_deep_ensemble_main, ensemble_state={}, f_save: bool = True):
        """
        Creates particles and launches push distribution training loop.
        
        :param Callable dataloader: Dataloader.
        :param int epochs: Number of epochs to train for. 
        :param Callable loss_fn: Loss function to be used during training.
        :param int num_ensembles: The number of models to be ensembled.
        :param any mk_optim: Returns an optimizer.
        :param function ensemble_entry: Training loop for deep ensemble.
        :param dict ensemble_state: a dictionary to store state variables for ensembled models. i.e. in swag we need to know how
           how many swag epochs have passed to properly calculate a running average of model weights.
        :param bool f_save: Flag to save each particle/model. Requires "particles" folder in root directory of the script calling train_deep_ensemble


        """
        # 1. Create particles
        pids = [
            self.push_dist.p_create(mk_optim, device=(0 % self.num_devices), receive={
                "ENSEMBLE_MAIN": ensemble_entry
            }, state=ensemble_state)]
        for n in range(1, num_ensembles):
            pids += [self.push_dist.p_create(mk_optim, device=(n % self.num_devices), receive={
                "ENSEMBLE_STEP": _ensemble_step,
            }, state={})]

        # 2. Perform independent training
        self.push_dist.p_wait([self.push_dist.p_launch(0, "ENSEMBLE_MAIN", dataloader, loss_fn, epochs)])

        if f_save:
            self.push_dist.save()

# =============================================================================
# Deep Ensemble Training
# =============================================================================

def train_deep_ensemble(dataloader: Callable, loss_fn: Callable, epochs: int,
                        nn: Callable, *args, num_devices: int = 1, cache_size: int = 4, view_size: int = 4,
                        num_ensembles: int = 2, mk_optim = mk_optim,
                        ensemble_entry = _deep_ensemble_main, ensemble_state={}) -> None:
    """
    Returns a list of model paramters for a deep ensemble.

    
    :param Callable dataloader: Dataloader.
    :param Callable loss_fn: Loss function to be used during training.
    :param int epochs: Number of epochs to train for.
    :param Callable nn: The base model to be ensembled and trained.
    :param any *args: Any arguments needed for the model's initialization.
    :param int num_devices: The desired number of gpu devices to be utilized during training.
    :param int cache_size: The desired size of cache allocated to storing particles.
    :param int view_size: The number of other particle's parameters that can be seen by a particle on a single GPU.
    :param int num_ensembles: The number of models to be ensembled.
    :param any mk_optim: Returns an optimizer.
    :param function ensemble_entry: Training loop for deep ensemble.
    :param dict ensemble_state: a dictionary to store state variables for ensembled models. i.e. in swag we need to know how
           how many swag epochs have passed to properly calculate a running average of model weights.
    """
    ensemble = Ensemble(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    ensemble.bayes_infer(dataloader, epochs, loss_fn=loss_fn, num_ensembles=num_ensembles, mk_optim=mk_optim,
                         ensemble_entry=ensemble_entry, ensemble_state=ensemble_state)
    return ensemble.p_parameters()
