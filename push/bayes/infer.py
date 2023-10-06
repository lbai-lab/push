import atexit
import torch
from torch.utils.data import DataLoader
from typing import *

import push.push as ppush


class Infer:
    """Base inference class
    """    
    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        self.mk_nn = mk_nn
        self.args = args
        self.num_devices = num_devices
        self.cache_size = cache_size
        self.view_size = view_size
        
        # Create a PusH Distribution
        self.push_dist = ppush.PusH(self.mk_nn, *self.args, cache_size=self.cache_size, view_size=self.view_size)
        atexit.register(self._cleanup)

    def bayes_infer(self, dataloader: DataLoader, epochs: int, **kwargs) -> None:
        """infer function"""
        raise NotImplementedError
    
    def p_parameters(self) -> List[List[torch.Tensor]]:
        """returns parameters"""
        return [self.push_dist.p_parameters(pid) for pid in self.push_dist.particle_ids()]

    def _cleanup(self):
        """cleanup functions"""
        self.push_dist._cleanup()

    def __enter__(self):
        """enter function"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """exit function"""
        self.push_dist.__exit__(exc_type, exc_value, traceback)
