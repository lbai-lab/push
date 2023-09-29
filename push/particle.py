import torch
import torch.nn as nn
from typing import *

from push.lib.utils import detach_to_device, to_device
from push.lib.waitable import Waitable
from push.pfuture import PFuture


class Particle(Waitable):
    """User-facing particle interface.

    Implements concurrent particles.
    """
    def __init__(self, node_event_loop, device: int, pid: int, module: nn.Module, state: Dict[str, any]):
        assert isinstance(device, int)
        assert isinstance(pid, int)
        assert isinstance(module, nn.Module)

        self.device = device                     # Device the particle is on
        self.module = module                     # The module corresponding to the particle
        self.pid = pid                           # The particles identifier
        self.state = state                       # Particle's state

        # NOTE: WARNING: DO NOT ACCESS!!!
        self._node_event_loop = node_event_loop  # The device event loop that the particle is part of

    # -----------------------------------------------------
    # Particle functionality
    # -----------------------------------------------------

    def particle_ids(self) -> List[int]:
        """Returns all particles.

        Returns:
            List[int]: List of all particle identifiers visible to current particle.
        """
        return self._node_event_loop.particles()

    def register_receive(self, msg: str, fn: Callable, state: dict[str, any]) -> None:
        """Register receive functionality for current particle.

        Args:
            msg (str): Message for current particle to respond to.
            fn (Callable): Function to execute on `msg`.
            state (dict[str, any]): User state.

        Returns:
            None.
        """        
        return self._node_event_loop.register_receive(self.pid, msg, fn, state)

    def send(self, pid: int, msg: str, *args: any) -> PFuture:   
        """Send a `msg` to `pid` with arguments `*args` from current particle.

        Args:
            pid (int): Particle identifier of receiver.
            msg (str): Message.
            *args (any): Any additional arguments to give to receiver.

        Returns:
            PFuture: Handle to future result.
        """        
        return self._node_event_loop.send(self, self.pid, pid, msg, *args)

    def get(self, pid: int) -> PFuture:
        """Obtains particle `pid`'s parameters.

        Args:
            pid (int): Particle identifier to obtain parameters of.

        Returns:
            PFuture: Handle to future result.
        """        
        return self._node_event_loop.get(self.pid, pid)

    def wait(self, pfutures: List[PFuture]) -> List[any]:
        """Pause execution of current particle until all `pfutures` have been resolved.

        Args:
            pfutures (List[PFuture]): Future values to wait on.

        Returns:
            List[any]: List of resolved future values.
        """        
        return self._node_event_loop.wait(pfutures)

    # -----------------------------------------------------
    # NN functionality
    # -----------------------------------------------------

    def zero_grad(self) -> PFuture:
        """Zeros gradients on current particle.

        Returns:
            PFuture: Wait ensures that the gradients have been zeroed.
        """        
        self._node_event_loop.zero_grad(self.pid)

    def forward(self, x: torch.Tensor, *args: any) -> PFuture:
        """Performs a forward pass.

        Args:
            x (torch.Tensor): Input to the particle.

        Returns:
            PFuture: Future that eventually contains the value of the forward pass.
        """        
        x = detach_to_device(self.device, x, requires_grad=True)
        args = [detach_to_device(self.device, arg, requires_grad=True) for arg in args]
        # x = to_device(self.device, x, requires_grad=True)
        # args = [to_device(self.device, arg, requires_grad=True) for arg in args]
        return self._node_event_loop.forward(self.pid, x, *args)
        
    def step(self, loss_fn: Callable, data: torch.Tensor, label: torch.Tensor, *args) -> PFuture:
        """Performs a forward and backward pass using the registered optimizer.

        Args:
            loss_fn (Callable): Loss function to take a step with respect to.
            data (torch.Tensor): Data.
            label (torch.Tensor): label.

        Returns:
            PFuture: Future that eventually contains the loss of the step.
        """        
        data = detach_to_device(self.device, data)
        label = detach_to_device(self.device, label)
        args = [detach_to_device(self.device, arg) for arg in args]
        return self._node_event_loop.step(self.pid, loss_fn, data, label, *args)


class ParticleView:
    """User-facing particle view interface.

    Enables you to view another particle's parameters.
    """
    def __init__(self, view_cache, pid: int):
        self._view_cache = view_cache
        self.pid = pid

    def view(self) -> nn.Module:
        """Call to access it's latest state.

        Returns:
            nn.Module: The other particle.
        """        
        return self._view_cache.try_read(self.pid)
