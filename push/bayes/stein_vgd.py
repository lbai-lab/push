import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *


from push.bayes.infer import Infer
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like


# =============================================================================
# Helper
# =============================================================================

def mk_empty_optim(params):
    # Limitiation must be global
    """

    Returns nothing

    """
    return None


# -----------------------------------------------------
# Prior
# -----------------------------------------------------
    
def normal_prior(params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """
    Returns the gradients of a set of tensors with respect to a normal distribution.

    This function iterates over the provided tensors, calculates the log probability of each tensor under a normal distribution
    with mean 0.0 and standard deviation 1.0. It then computes and appends the gradients with respect to each parameter.
    The resulting list contains the gradient tensors.
    
    :param Iterable[torch.Tensor] params: A collection of tensors whose gradients will be calculated.
    :return A list containing the computed gradients for each parameter.
    :rtype list[torch.Tensor]

    """
    normal = Normal(0.0, 1.0)
    grads = []
    for param in params:
        # TODO: change me for different NNs
        y = torch.sum(normal.log_prob(param.requires_grad_()))
        grads += [torch.autograd.grad(y, param, grad_outputs=torch.ones_like(y), create_graph=True,)[0]]
    return grads


# -----------------------------------------------------
# SVGD Kernels
# -----------------------------------------------------

def torch_squared_exp_kernel(x: torch.Tensor, y: torch.Tensor, length_scale: float) -> torch.Tensor:
    """
    Computes the squared exponential kernel between two tensors.

    This function calculates the squared exponential kernel value between two tensors `x` and `y`. 
    The characteristic length scale of the kernel is specified by `length_scale`.
    The squared exponential kernel is given by exp(-0.5 * (x - y)^2 / length_scale^2).

    :param torch.Tensor x: The first input tensor.
    :param torch.Tensor y: The second input tensor.
    :param float length_scale: The characteristic length scale of the kernel.

    :return The computed squared exponential kernel value.
    :rtype torch.Tensor

    :note This kernel is commonly used in Gaussian Process regression for modeling smooth functions.

    """
    diff = (x - y) / length_scale
    radius2 = torch.dot(diff, diff)
    return torch.exp(-0.5 * radius2)

def torch_squared_exp_kernel_grad(x: torch.Tensor, y: torch.Tensor, length_scale: float) -> torch.Tensor:
    """
    Computes the gradient of the squared exponential kernel with respect to its inputs.

    This function calculates the gradient of the squared exponential kernel with respect to its inputs, 
    `x` and `y`, as well as with respect to the characteristic length scale `length_scale`.

    :param torch.Tensor x: The first input tensor.
    :param torch.Tensor y: The second input tensor.
    :param float length_scale: The characteristic length scale of the kernel.
    :return The computed gradient of the squared exponential kernel.
    :rtype torch.Tensor

    """
    prefactor = (x - y) / (length_scale ** 2)
    return -prefactor * torch_squared_exp_kernel(x, y, length_scale)


# =============================================================================
# SVGD
# =============================================================================

def _svgd_leader(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs) -> None:
    # TODO comment explaining this function
    n = len(particle.particle_ids())
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

    for e in tqdm(range(epochs)):
        losses = []
        for data, label in dataloader:
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
                    update[pid1] += torch_squared_exp_kernel(p_j, p_i, lengthscale) * p_j_grad
                    update[pid1] += torch_squared_exp_kernel_grad(p_j, p_i, lengthscale)
                update[pid1] = update[pid1] / n

            # 4. Send kernel
            futs = [particle.send(pid, "SVGD_FOLLOW", lr, update[pid]) for pid in other_particles]
            [f.wait() for f in futs]
            _svgd_follow(particle, lr, update[particle.pid])

            loss = loss_fn(particle.forward(data).wait().to("cpu"), label)
            losses += [torch.mean(torch.tensor(loss))]

        print(f"Average loss {torch.mean(torch.tensor(losses))}")


def _svgd_leader_memeff(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs) -> None:
    # TODO comment explaining this function
    n = len(particle.particle_ids())
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

    for e in tqdm(range(epochs)):
        losses = []
        for data, label in dataloader:
            # 1. Step every particle
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
                    update.add_(p_j_grad, alpha=k)
                    update.add_(diff)
                    # update += torch_squared_exp_kernel(p_j, p_i, lengthscale) * p_j_grad
                    # update += torch_squared_exp_kernel_grad(p_j, p_i, lengthscale)
                update = update / n
                return update

            # 3. Compute kernel and kernel gradients
            for pid1, params1 in particles.items():
                # 4. Send kernel
                if pid1 != particle.pid:
                    update = compute_update(pid1, params1)
                    particle.send(pid, "SVGD_FOLLOW", lr, update).wait()
            update = compute_update(particle.pid, particles[particle.pid])
            _svgd_follow(particle, lr, update)
            if particle.state["args"].model == "fno":
                (_data, grid) = data
                loss = loss_fn(particle.forward(_data, grid ,label).wait().to("cpu"), label)
            else:
                loss = loss_fn(particle.forward(data).wait().to("cpu"), label)

            losses += [torch.mean(torch.tensor(loss))]

        print(f"Average loss {torch.mean(torch.tensor(losses))}")


def _svgd_step(particle: Particle, loss_fn: Callable, data: torch.Tensor, label: torch.Tensor) -> None:
    # TODO comment explaining this function
    return particle.step(loss_fn, data, label)


def _svgd_follow(particle: Particle, lr: float, update: List[torch.Tensor]) -> None:
    # TODO comment explaining this function
    # 1. Unflatten
    params = list(particle.module.parameters())
    updates = unflatten_like(update.unsqueeze(0), params)
    
    # 2. Apply the update to the input particle
    with torch.no_grad():
        for p, up in zip(params, updates):
            p.add_(up.real, alpha=-lr)


class SteinVGD(Infer):
    """SteinVGD Class.

    This class extends the 'Infer' class and uses Stein Variational Gradient Descent (SteinVGD) 
    for Bayesian inference tasks.

    Args:
        mk_nn (Callable): A function that creates the neural network architecture for the model.
        *args (any): Additional arguments that will be passed to the 'Infer' class.
        num_devices (int): The number of devices to be used for computation. Default is 1.
        cache_size (int): The size of the cache for storing computed gradients. Default is 4.
        view_size (int): The size of the view for distributed computations. Default is 4.
    """

    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        super(SteinVGD, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        
    def bayes_infer(self,
                    dataloader: DataLoader, epochs: int,
                    prior=None, loss_fn=torch.nn.MSELoss(),
                    num_particles=1, lengthscale=1.0, lr=1e-3,
                    svgd_entry=_svgd_leader, svgd_state={}):
        """Performs Bayesian inference using SteinVGD.

        This method performs Bayesian inference using Stein Variational Gradient Descent (SteinVGD).

        Overridden from the `Infer` class.

        Args:
            dataloader (DataLoader): The dataloader to use for training.
            epochs (int): The number of epochs to train for.
            prior (Callable, optional): Prior information for Bayesian inference. Default is None.
            loss_fn (Callable, optional): The loss function to be used during training. Default is torch.nn.MSELoss().
            num_particles (int, optional): The number of particles to use in SVGD. Default is 1.
            lengthscale (float, optional): The characteristic length scale of the SVGD kernel. Default is 1.0.
            lr (float, optional): The learning rate for optimization. Default is 1e-3.
            svgd_entry (Callable, optional): The SVGD entry function. Default is _svgd_leader.
            svgd_state (dict, optional): Additional state information for SVGD. Default is {}.

        """
        pid_leader = self.push_dist.p_create(mk_empty_optim, device=0, receive={
            "SVGD_LEADER": svgd_entry
        }, state=svgd_state)
        pids = [pid_leader]
        for p in range(num_particles - 1):
            pid = self.push_dist.p_create(mk_empty_optim, device=((p + 1) % self.num_devices), receive={
                "SVGD_STEP": _svgd_step,
                "SVGD_FOLLOW": _svgd_follow
            }, state=svgd_state)
            pids += [pid]
        self.push_dist.p_wait([self.push_dist.p_launch(0, "SVGD_LEADER", prior, loss_fn, lengthscale, lr, dataloader, epochs)])


# =============================================================================
# SVGD Train
# =============================================================================

def train_svgd(dataloader: DataLoader, loss_fn: Callable, epochs: int, num_particles: int, nn: Callable, *args,
               lengthscale=1.0, lr=1e3, prior=None,
               num_devices=1, cache_size=4, view_size=4,
               svgd_entry=_svgd_leader, svgd_state={}) -> None:
    """
    Trains a model using Stein Variational Gradient Descent (SVGD).

    This function trains a model using Stein Variational Gradient Descent (SVGD). It initializes a `SteinVGD` instance
    and performs Bayesian inference using the provided data loader, loss function, and training parameters. The resulting
    parameters from SVGD are returned.
    
    Args:
        dataloader (DataLoader): The data loader for the training data.
        loss_fn (Callable): The loss function to be used during training.
        epochs (int): The number of training epochs.
        num_particles (int): The number of particles to use in SVGD.
        nn (Callable): A function that creates the neural network architecture for the model.
        *args (any): Additional arguments to be passed to the `SteinVGD` constructor.
        lengthscale (float, optional): The characteristic length scale of the SVGD kernel. Default is 1.0.
        lr (float, optional): The learning rate for optimization. Default is 1e3.
        prior: Prior information for Bayesian inference. Default is None.
        num_devices (int, optional): The number of devices to be used for computation. Default is 1.
        cache_size (int, optional): The size of the cache for storing computed gradients. Default is 4.
        view_size (int, optional): The size of the view for distributed computations. Default is 4.
        svgd_entry (Callable, optional): The SVGD entry function. Default is `_svgd_leader`.
        svgd_state (dict, optional): Additional state information for SVGD. Default is {}.

    Returns:
        None

    Note:
        The returned parameters can be used for further inference, testing, and analysis.
    """
    with SteinVGD(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size) as stein_vgd:
        stein_vgd.bayes_infer(dataloader, epochs,
                              prior=prior, loss_fn=loss_fn,
                              num_particles=num_particles, lengthscale=lengthscale, lr=lr,
                              svgd_entry=svgd_entry, svgd_state=svgd_state)
        return stein_vgd.p_parameters()
