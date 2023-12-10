import torch

from typing import Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
from push.bayes.infer import Infer
from push.particle import Particle
from dropout_util import FixableDropout, patch_dropout
from push.lib.utils import detach_to_cpu

# =============================================================================
# Helper
# =============================================================================

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
    # Training loop
    for e in tqdm(range(epochs)):
        losses = []
        for data, label in dataloader:
            loss = particle.step(loss_fn, data, label).wait()
            losses += [loss]
            for pid in particle.other_particles():
                particle.send(pid, "MULTIMC_STEP", loss_fn, data, label)

def _multimc_step(particle: Particle):
    particle.step()
    
# =============================================================================
# MC Dropout Inference
# =============================================================================

def _leader_pred_dl(particle: Particle, dataloader: DataLoader, f_reg: bool = True, mode="mean") -> torch.Tensor:
    acc = []
    for data, label in dataloader:
        acc += [_leader_pred(particle, data, f_reg=f_reg, mode=mode)]
    return torch.cat(acc)

def _leader_pred(particle: Particle, data: torch.Tensor, f_reg: bool = True, mode="mean") -> torch.Tensor:
    preds = []
    preds += [detach_to_cpu(particle.forward(data).wait())]
    for pid in particle.other_particles():
        preds += [particle.send(pid, "ENSEMBLE_PRED", data).wait()]
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
        else:
            raise ValueError(f"Mode {mode} not supported ...")
    else:
        cls = t_preds.softmax(dim=1).argmax(dim=1)
        return torch.mode(cls, dim=1)

def _multimc_pred(particle: Particle):
    pass

# =============================================================================
# Multi MC Dropout
# =============================================================================
class MultiMCDropout(Infer):
    """Multi MC Dropout inference
    
    
    """
    def __init__(self, mk_nn: Callable[..., Any], *args: any, patch=True, num_devices=1, cache_size=4, view_size=4) -> None:
        def mk_module(*args):
            if patch:
                return patch_dropout(mk_nn(*args))
        super(MultiMCDropout, self).__init__(mk_module, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)

    def bayes_infer(self,
                    dataloader: DataLoader, 
                    epochs: int,
                    loss_fn: Callable = torch.nn.CrossEntropyLoss(),
                    size_ensemble: int = 2,
                    mk_optim=mk_optim, f_save: bool = False):
        """Create PusH distribution and train it.
        
        """
        # Create ensemble
        pids = []
        pid_main = self.push_dist.p_create(mk_optim, device=0, receive={
            "MULTIMC_MAIN": _multimc_main,
            "LEADER_PRED_DL": _leader_pred_dl,
            "LEADER_PRED": _leader_pred,
        })
        pids.append(pid_main)
        for i in range(size_ensemble-1):
            pids.append(self.push_dist.p_create(mk_optim, device=i+1, receive={
                "MULTIMC_STEP": _multimc_step,
                "MULTIMC_PRED": _multimc_pred,
            }))

        # Train ensemble
        self.push_dist.p_wait([self.push_dist.p_launch(pid_main, "MULTIMC_MAIN", dataloader, loss_fn, epochs)])

        # Save ensemble
        if f_save:
            self.push_dist.save()

    def posterior_pred(self, data: DataLoader | torch.Tensor, f_reg=True, mode="mean") -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            pass
        if isinstance(data, DataLoader):
            pass
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")