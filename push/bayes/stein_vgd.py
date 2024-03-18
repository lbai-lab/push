import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *


from push.bayes.infer import Infer
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like
from push.bayes.ensemble import _leader_pred_dl, _leader_pred, _ensemble_pred
import torch.optim.lr_scheduler as lr_scheduler



# =============================================================================
# Helper
# =============================================================================

def mk_empty_optim(params):
    """
    Helper function to create an empty optimizer.

    Args:
        params: Model parameters.

    Returns:
        None.

    """
    return None

def mk_empty_scheduler(optim):
    """
    Helper function to create an empty optimizer.

    Args:
        params: Model parameters.

    Returns:
        None.

    """
    return None

# -----------------------------------------------------
# Prior
# -----------------------------------------------------

def normal_prior(params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute gradients with respect to a normal distribution.

    This function calculates the gradients with respect to a normal distribution with mean 0.0 and standard deviation 1.0.

    Args:
        params (Iterable[torch.Tensor]): Collection of tensors for which gradients are computed.

    Returns:
        List[torch.Tensor]: List of computed gradients for each parameter.

    """
    normal = Normal(0.0, 1.0)
    grads = []
    for param in params:
        y = torch.sum(normal.log_prob(param.requires_grad_()))
        grads += [torch.autograd.grad(y, param, grad_outputs=torch.ones_like(y), create_graph=True,)[0]]
    return grads


# -----------------------------------------------------
# SVGD Kernels
# -----------------------------------------------------

def torch_squared_exp_kernel(x: torch.Tensor, y: torch.Tensor, length_scale: float) -> torch.Tensor:
    """
    Compute the squared exponential kernel between two tensors.

    This function calculates the squared exponential kernel value between two tensors `x` and `y`.
    The kernel has a characteristic length scale specified by `length_scale`.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        length_scale (float): Characteristic length scale of the kernel.

    Returns:
        torch.Tensor: Computed squared exponential kernel value.

    Note:
        The kernel is commonly used in Gaussian Process regression for modeling smooth functions.

    """
    diff = (x - y) / length_scale
    radius2 = torch.dot(diff, diff)
    return torch.exp(-0.5 * radius2)

def torch_squared_exp_kernel_grad(x: torch.Tensor, y: torch.Tensor, length_scale: float) -> torch.Tensor:
    """
    Compute the gradient of the squared exponential kernel.

    This function calculates the gradient of the squared exponential kernel with respect to its inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        length_scale (float): Characteristic length scale of the kernel.

    Returns:
        torch.Tensor: Computed gradient of the squared exponential kernel.

    """
    prefactor = (x - y) / (length_scale ** 2)
    return -prefactor * torch_squared_exp_kernel(x, y, length_scale)


# =============================================================================
# SVGD
# =============================================================================

def _svgd_leader(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs, bootstrap = False) -> None:
    """
    Perform SVGD update for the leader particle.

    Args:
        particle (Particle): Leader particle being operated on.
        prior: Prior information for Bayesian inference.
        loss_fn (Callable): Loss function to be used during training.
        lengthscale: Characteristic length scale of the SVGD kernel.
        lr (float): Learning rate for optimization.
        dataloader (DataLoader): Dataloader for training data.
        epochs (int): Number of training epochs.

    """

    if bootstrap:
        print("yet to be implemented")
    else:
        n = len(particle.particle_ids())
        other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

        for e in tqdm(range(epochs)):
            losses = []
            for data, label in dataloader:
                fut = particle.step(loss_fn, data, label)
                futs = [particle.send(pid, "SVGD_STEP", loss_fn, data, label) for pid in other_particles]
                fut.wait(); [f.wait() for f in futs]

                particles = {pid: (particle.get(pid) if pid != particle.pid else
                    list(particle.module.parameters())) for pid in particle.particle_ids()}
                for pid in other_particles:
                    particles[pid] = particles[pid].wait()

                update = {}
                for pid1, params1 in particles.items():
                    params1 = list(particles[pid1].view().parameters()) if pid1 != particle.pid else params1
                    p_i = flatten(params1)
                    update[pid1] = torch.zeros_like(p_i)
                    for pid2, params2 in particles.items():
                        params2 = list(particles[pid2].view().parameters()) if pid2 != particle.pid else params2

                        p_j = flatten(params2)
                        p_j_grad = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in params2])
                        update[pid1] += torch_squared_exp_kernel(p_j, p_i, lengthscale) * p_j_grad
                        update[pid1] += torch_squared_exp_kernel_grad(p_j, p_i, lengthscale)
                    update[pid1] = update[pid1] / n

                futs = [particle.send(pid, "SVGD_FOLLOW", lr, update[pid]) for pid in other_particles]
                [f.wait() for f in futs]
                _svgd_follow(particle, lr, update[particle.pid])
                loss = loss_fn(particle.forward(data).wait().to("cpu"), label)
                losses += [torch.mean(loss).item()]
            print(f"Average loss {torch.mean(torch.tensor(losses))}")


def _svgd_leader_memeff(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs) -> None:
    """
    Perform SVGD update for the leader particle with memeff kernel computation.

    Args:
        particle (Particle): Leader particle being operated on.
        prior: Prior information for Bayesian inference.
        loss_fn (Callable): Loss function to be used during training.
        lengthscale: Characteristic length scale of the SVGD kernel.
        lr (float): Learning rate for optimization.
        dataloader (DataLoader): Dataloader for training data.
        epochs (int): Number of training epochs.

    """
    n = len(particle.particle_ids())
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))

    for e in tqdm(range(epochs)):
        losses = []
        for data, label in dataloader:
            fut = particle.step(loss_fn, data, label)
            futs = [particle.send(pid, "SVGD_STEP", loss_fn, data, label) for pid in other_particles]
            fut.wait(); [f.wait() for f in futs]

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

                    p_j = flatten(params2)
                    p_j_grad = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in params2])
                    diff = (p_j - p_i) / lengthscale
                    radius2 = torch.dot(diff, diff)
                    k = torch.exp(-0.5 * radius2).item()
                    diff.mul_(-k / lengthscale)
                    update.add_(p_j_grad, alpha=k)
                    update.add_(diff)
                update = update / n
                return update

            for pid1, params1 in particles.items():
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

        # print(f"Average loss {torch.mean(torch.tensor(losses))}")


def _svgd_step(particle: Particle, loss_fn: Callable, data: torch.Tensor, label: torch.Tensor) -> None:
    """
    Perform a single step of SVGD update.

    Args:
        particle (Particle): Particle being operated on.
        loss_fn (Callable): Loss function to be used during training.
        data (torch.Tensor): Input data tensor.
        label (torch.Tensor): Target label tensor.

    """
    return particle.step(loss_fn, data, label)


def _svgd_follow(particle: Particle, lr: float, update: List[torch.Tensor]) -> None:
    """
    Update particle parameters based on SVGD kernel information.

    Args:
        particle (Particle): Particle being operated on.
        lr (float): Learning rate for optimization.
        update (List[torch.Tensor]): Update tensor.

    """
    params = list(particle.module.parameters())
    updates = unflatten_like(update.unsqueeze(0), params)

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
        """
        Initialize the SteinVGD model.

        Args:
            mk_nn (Callable): Function to create the neural network architecture.
            *args (any): Additional arguments to be passed to the 'Infer' class.
            num_devices (int): Number of devices to be used for computation. Default is 1.
            cache_size (int): Size of the cache for storing computed gradients. Default is 4.
            view_size (int): Size of the view for distributed computations. Default is 4.

        """
        super(SteinVGD, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)

    def bayes_infer(self,
                    dataloader: DataLoader, epochs: int,
                    prior=False, random_seed=False, bootstrap = False, loss_fn=torch.nn.MSELoss(),
                    num_particles=1, lengthscale=1.0, lr=1e-3,
                    svgd_entry=_svgd_leader, svgd_state={}):
        """
        Perform Bayesian inference using SteinVGD.

        Args:
            dataloader (DataLoader): Dataloader for training.
            epochs (int): Number of training epochs.
            prior: Prior information for Bayesian inference. Default is None.
            loss_fn (Callable): Loss function to be used during training. Default is torch.nn.MSELoss().
            num_particles (int): Number of particles to use in SVGD. Default is 1.
            lengthscale (float): Characteristic length scale of the SVGD kernel. Default is 1.0.
            lr (float): Learning rate for optimization. Default is 1e-3.
            svgd_entry (Callable): SVGD entry function. Default is _svgd_leader.
            svgd_state (dict): Additional state information for SVGD. Default is {}.

        """
        if random_seed:
            train_keys = torch.randint(0, int(1e9), (num_particles,), dtype=torch.int64).tolist()
        else:
            train_keys = [None] * num_particles
        pid_leader = self.push_dist.p_create(mk_empty_optim, mk_scheduler=mk_empty_scheduler, prior=prior, train_key=train_keys[0], device=0, receive={
            "SVGD_LEADER": svgd_entry,
            "LEADER_PRED_DL": _leader_pred_dl,
            "LEADER_PRED": _leader_pred,
        }, state=svgd_state)
        pids = [pid_leader]
        for p in range(num_particles - 1):
            pid = self.push_dist.p_create(mk_empty_optim, mk_scheduler=mk_empty_scheduler, prior=prior, train_key=train_keys[p], device=((p + 1) % self.num_devices), receive={
                "SVGD_STEP": _svgd_step,
                "SVGD_FOLLOW": _svgd_follow,
                "ENSEMBLE_PRED": _ensemble_pred,
            }, state=svgd_state)
            pids += [pid]
        self.push_dist.p_wait([self.push_dist.p_launch(0, "SVGD_LEADER", prior, loss_fn, lengthscale, lr, dataloader, epochs, bootstrap)])

    def posterior_pred(self, data: DataLoader, f_reg=True, mode=["mean"]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            fut = self.push_dist.p_launch(0, "LEADER_PRED", data, f_reg, mode)
            return self.push_dist.p_wait([fut])[fut._fid]
        elif isinstance(data, DataLoader):
            fut = self.push_dist.p_launch(0, "LEADER_PRED_DL", data, f_reg, mode)
            return self.push_dist.p_wait([fut])[fut._fid]
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")


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
    stein_vgd = SteinVGD(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    stein_vgd.bayes_infer(dataloader, epochs,
                          prior=prior, loss_fn=loss_fn,
                          num_particles=num_particles, lengthscale=lengthscale, lr=lr,
                          svgd_entry=svgd_entry, svgd_state=svgd_state)
    return stein_vgd
