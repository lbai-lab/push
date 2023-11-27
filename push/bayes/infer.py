import atexit
import torch
from torch.utils.data import DataLoader
from typing import *

import push.push as ppush


class Infer:
    """Base Infer class
    
    Creates a PusH distribution with an inference method and return parameters method.

    Infer is a base class that should be inherited by a child class that implements a Bayesian inference method.

    Args:
        mk_nn (Callable): Function to create base model.
        *args (any): Any arguments required for base model to be initialized.
        num_devices (int, optional): The desired number of gpu devices that will be utilized. Defaults to 1.
        cache_size (int, optional): The size of cache used to store particles. Defaults to 4.
        view_size (int, optional): The number of particles to consider storing in cache. Defaults to 4.
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
        """Bayesian inference method.

        This method should be overridden by a child class.

        Args:
            dataloader (DataLoader): The dataloader to use for training.
            epochs (int): The number of epochs to train for.

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError
    
    def p_parameters(self) -> List[List[torch.Tensor]]:
        """Return parameters of all particles.
        
        Returns:
            List[List[torch.Tensor]]: List of all particle parameters.
        """
        return [self.push_dist.p_parameters(pid) for pid in self.push_dist.particle_ids()]

    def _cleanup(self):
        self.push_dist._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.push_dist.__exit__(exc_type, exc_value, traceback)