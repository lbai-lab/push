import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *

from push.bayes.dropout_util import patch_dropout
from push.bayes.infer import Infer
from push.particle import Particle
from push.lib.utils import detach_to_cpu


# =============================================================================
# Helper
# =============================================================================

def mk_module(mk_nn: Callable[..., Any], patch, *args: any) -> nn.Module:
    """Wrapper for mk_nn that patches dropout if set.

    NOT meant to be used outside of MultiMCDropout.

    Args:
        mk_nn (Callable): Function that returns a nn.Module.
        patch (bool): Whether to patch dropout layers.
        *args (list): Arguments for mk_nn.
    
    Returns:
        nn.Module: Patched nn.Module.
    """
    if patch:
        return mk_nn(*args).apply(patch_dropout)
    else:
        return mk_nn(*args)


def mk_optim(params):
    """Returns Adam optimizer.
    
    Args:
        params: Model parameters.
    
    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    return torch.optim.Adam(params, lr=1e-4, weight_decay=1e-2)


# =============================================================================
# MC Dropout Particle Functions
# =============================================================================

def _multimc_main(particle: Particle, dataloader: DataLoader, loss_fn: Callable, epochs: int) -> None:
    print_loop = max(int(epochs/10),1)
    tq = tqdm(range(epochs))
    # Training loop
    for e in tq:
        losses_futs = []
        for data, label in dataloader:
            losses_futs += [particle.step(loss_fn, data, label)]
            for pid in particle.other_particles():
                losses_futs += [particle.send(pid, "MULTIMC_STEP", loss_fn, data, label)]
        if e % print_loop == 0:
            losses = [l.wait() for l in losses_futs]
            average_loss = sum(losses)/len(losses)
            tq.set_postfix({"loss: " : average_loss})
            # print(f"Average loss after epoch {e}: {average_loss}")


def _multimc_step(particle: Particle, loss_fn: Callable, data, label, *args):
    particle.module.train()
    return particle.step(loss_fn, data, label, *args).wait()


# =============================================================================
# MC Dropout Inference
# =============================================================================

def freeze_dropout(particle: Particle):
    # Set FixableDropout to eval mode
    particle.module.eval()

    # Set normal (unpatched) dropout layers to train mode
    for module in particle.module.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def _leader_pred_dl(particle: Particle, dataloader: DataLoader, f_reg=True, mode="mean", num_samples=10, freeze_on_eval=True) -> torch.Tensor:
    acc = []
    for data, label in dataloader:
        acc += [_leader_pred(particle, data, f_reg=f_reg, mode=mode)]
    return torch.cat(acc)


def _leader_pred(particle: Particle, data, f_reg=True, mode="mean", num_samples=10, freeze_on_eval=True):
    if freeze_on_eval:
        freeze_dropout(particle)
    else:
        particle.module.train()
    preds = []
    my_preds = [detach_to_cpu(particle.forward(data).wait()) for _ in range(num_samples)]
    preds += torch.stack(my_preds, dim=0)
    for pid in particle.other_particles():
        preds += particle.send(pid, "MULTIMC_PRED", data, num_samples).wait()
    t_preds = torch.stack(preds, dim=1)
    if f_reg:
        if mode == "mean":
            return t_preds.mean(dim=1)
        elif mode == "median":
            return t_preds.median(dim=1).values
        elif mode == "min":
            return t_preds.min(dim=1).values
        elif mode == "max":
            return t_preds.max(dim=1).values
        elif mode == "raw":
            return t_preds
        else:
            raise ValueError(f"Mode {mode} not supported ...")
    else:
        cls = t_preds.softmax(dim=1).argmax(dim=1)
        return torch.mode(cls, dim=1)


def _multimc_pred(particle: Particle, data: torch.Tensor, num_samples: int = 10, freeze_on_eval: bool = True):
    if freeze_on_eval:
        freeze_dropout(particle)
    else:
        particle.module.train()
    preds = [detach_to_cpu(particle.forward(data).wait()) for _ in range(num_samples)]
    return torch.stack(preds, dim=0)


# =============================================================================
# Multi MC Dropout
# =============================================================================

class MultiMCDropout(Infer):
    """Multi MC Dropout Class.

    5hether to patch dropout layers.
        num_devices (int): Number of devices to use.
        cache_size (int): Size of cache.
        view_size (int): Size of view.
    """
    def __init__(self, mk_nn: Callable[..., Any], *args: any, patch=True, num_devices=1, cache_size=4, view_size=4) -> None:
        super(MultiMCDropout, self).__init__(mk_module, *(mk_nn, patch, *args), num_devices=num_devices, cache_size=cache_size, view_size=view_size)

    def bayes_infer(self,
                    dataloader: DataLoader, 
                    epochs: int,
                    loss_fn: Callable = torch.nn.CrossEntropyLoss(),
                    size_ensemble: int = 2,
                    mk_optim=mk_optim, mk_scheduler=None, f_save: bool = False):
        """Create PusH distribution and train it.
        """
        # Create main particle
        pids = []
        pid_main = self.push_dist.p_create(mk_optim, mk_scheduler, device=0, receive={
            "MULTIMC_MAIN": _multimc_main,
            "LEADER_PRED_DL": _leader_pred_dl,
            "LEADER_PRED": _leader_pred,
        })
        pids.append(pid_main)

        # Create worker particles
        for i in range(size_ensemble-1):
            pids.append(self.push_dist.p_create(mk_optim, mk_scheduler,device=(i % self.num_devices), receive={
                "MULTIMC_STEP": _multimc_step,
                "MULTIMC_PRED": _multimc_pred,
            }))

        # Train ensemble
        self.push_dist.p_wait([self.push_dist.p_launch(pid_main, "MULTIMC_MAIN", dataloader, loss_fn, epochs)])

        # Save ensemble
        if f_save:
            self.push_dist.save()

    def posterior_pred(self, data: DataLoader | torch.Tensor, f_reg=True, mode="mean", num_samples=10, freeze_on_eval=False) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            fut = self.push_dist.p_launch(0, "LEADER_PRED", data, f_reg, mode, num_samples, freeze_on_eval)
            return self.push_dist.p_wait([fut])[fut._fid]
        elif isinstance(data, DataLoader):
            fut = self.push_dist.p_launch(0, "LEADER_PRED_DL", data, f_reg, mode, num_samples, freeze_on_eval)
            return self.push_dist.p_wait([fut])[fut._fid]
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")

    @staticmethod
    def train_mc_dropout(dataloader: Callable, loss_fn: Callable, epochs: int,
                        nn: Callable, *args, num_devices: int = 1, cache_size: int = 4, view_size: int = 4,
                        size_ensemble: int = 2, mk_optim = mk_optim,
                        mc_entry = _multimc_main, patch=False) -> List[torch.Tensor]:
        """Creates and trains a MC Dropout ensemble.
        
        Args:
            dataloader (Callable): Dataloader.
            loss_fn (Callable): Loss function to be used during training.
            epochs (int): Number of epochs to train for.
            nn (Callable): Function that returns a nn.Module.
            *args (list): Arguments for nn.
            num_devices (int, optional): Number of devices to use.
            cache_size (int, optional): Size of cache.
            view_size (int, optional): Size of view.
            size_ensemble (int, optional): Number of models to be ensembled.
            mk_optim (Callable): Returns an optimizer.
            mc_entry (function, optional): Training loop for MC Dropout.
            patch (bool, optional): Whether to patch dropout layers.
        """
        mc_dropout = MultiMCDropout(nn, *args, patch=patch, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        mc_dropout.bayes_infer(dataloader, epochs, loss_fn, size_ensemble, mk_optim)
        return mc_dropout


# =============================================================================
# MC Dropout Training
# =============================================================================

def train_mc_dropout(*args, **kwargs):
    """Train a MC Dropout ensemble.
    
    Wraps MultiMCDropout.train_mc_dropout.
    """
    return MultiMCDropout.train_mc_dropout(*args, **kwargs)
