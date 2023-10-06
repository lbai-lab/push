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
    return None


# -----------------------------------------------------
# Prior
# -----------------------------------------------------
    
def normal_prior(params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """normal prior"""
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
    """torch  squared exp kernal"""
    diff = (x - y) / length_scale
    radius2 = torch.dot(diff, diff)
    return torch.exp(-0.5 * radius2)

def torch_squared_exp_kernel_grad(x: torch.Tensor, y: torch.Tensor, length_scale: float) -> torch.Tensor:
    """torch squared exp kernal grad"""
    prefactor = (x - y) / (length_scale ** 2)
    return -prefactor * torch_squared_exp_kernel(x, y, length_scale)


# =============================================================================
# SVGD
# =============================================================================

def _svgd_leader(particle: Particle, prior, loss_fn: Callable, lengthscale, lr, dataloader: DataLoader, epochs) -> None:
    """svgd leader"""
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
    """svgd leader memeff"""
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
    """svgd step"""
    return particle.step(loss_fn, data, label)


def _svgd_follow(particle: Particle, lr: float, update: List[torch.Tensor]) -> None:
    """svgd follow"""
    # 1. Unflatten
    params = list(particle.module.parameters())
    updates = unflatten_like(update.unsqueeze(0), params)
    
    # 2. Apply the update to the input particle
    with torch.no_grad():
        for p, up in zip(params, updates):
            p.add_(up.real, alpha=-lr)


class SteinVGD(Infer):
    """SteinVGD Class"""
    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        super(SteinVGD, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        
    def bayes_infer(self,
                    dataloader: DataLoader, epochs: int,
                    prior=None, loss_fn=torch.nn.MSELoss(),
                    num_particles=1, lengthscale=1.0, lr=1e-3,
                    svgd_entry=_svgd_leader, svgd_state={}):
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
    """train svgd"""
    with SteinVGD(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size) as stein_vgd:
        stein_vgd.bayes_infer(dataloader, epochs,
                              prior=prior, loss_fn=loss_fn,
                              num_particles=num_particles, lengthscale=lengthscale, lr=lr,
                              svgd_entry=svgd_entry, svgd_state=svgd_state)
        return stein_vgd.p_parameters()
