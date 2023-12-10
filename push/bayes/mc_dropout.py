from typing import Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
from push.bayes.infer import Infer
from push.particle import Particle
from dropout_util import FixableDropout, patch_dropout

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
# MC Dropout Functions
# =============================================================================

def _multimc_main(particle: Particle):
    pass

def _leader_pred_dl(particle: Particle):
    pass

def _leader_pred(particle: Particle):
    pass

def _multimc_step(particle: Particle):
    pass

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