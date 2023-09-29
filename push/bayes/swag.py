from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
import torch

from push.bayes.infer import Infer
from push.particle import Particle
import push.push as ppush
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu, to_device


# =============================================================================
# Helper
# =============================================================================

def mk_optim(params):
    return torch.optim.Adam(params, lr=1e-5, weight_decay=1e-3)


# =============================================================================
# Swag Training
# =============================================================================

def _swag_sample2(particle: Particle,
                  dataloader: DataLoader,
                  loss_fn: Callable,
                  scale: float,
                  var_clamp: float,
                  num_samples: int) -> None:
    """Inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    """
    pid = particle.pid
    # Gather
    mean_list = [param for name, param in particle.state[pid]["mom1"].items()]
    sq_mean_list = [param for name, param in particle.state[pid]["mom2"].items()]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # draw diagonal variance sample
    var = torch.clamp(sq_mean - mean ** 2, var_clamp)
    var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

    rand_sample = var_sample

    # Update sample with mean and scale
    sample = mean + scale_sqrt * rand_sample
    sample = sample.unsqueeze(0)

    # Update
    samples_list = unflatten_like(sample, mean_list)
    with torch.no_grad():
        for param, sample in zip(particle.module.parameters(), samples_list):
            param.copy_(sample)

    # 
    swag_losses = []
    for data, label in tqdm(dataloader):
        loss = loss_fn(detach_to_cpu(particle.forward(data).wait()), label)
        swag_losses += [loss.detach().to("cpu")]
    print("Average loss", particle.pid, torch.mean(torch.tensor(swag_losses)))


def _swag_step(particle: Particle,
               loss_fn: Callable,
               data: torch.Tensor,
               label: torch.Tensor,
               *args: any) -> None:
    particle.step(loss_fn, data, label,*args)


def update_theta(state, state_sq, param, param_sq, n):
    # for param in theta.keys():
    #     theta[param] = (theta[param]*n+tt[param])/(n+1)
    #     theta2[param] = (theta2[param]*n+tt2[param])/(n+1)
    for st, st_sq, p, p_sq in zip(state, state_sq, param, param_sq):
        st.data = (st.data * n + p.data)/(n+1)
        st_sq.data = (st_sq.data * n + p_sq.data)/(n+1)


def _swag_swag(particle: Particle, reset: bool) -> None:    
    state = particle.state
    if reset:
        state[particle.pid] = {
            "mom1": [param for param in particle.module.parameters()], # {name: param for name, param in particle.module.named_parameters()},
            "mom2": [param*param for param in particle.module.parameters()] # {name: param * param for name, param in particle.module.named_parameters()}
        }
    else:
        params = [param for param in particle.module.parameters()]  # {name: param for name, param in particle.module.named_parameters()}
        params_sq = [param*param for param in particle.module.parameters()]  # {name: param*param for name, param in particle.module.named_parameters()}
        update_theta(state[particle.pid]["mom1"], state[particle.pid]["mom2"], params, params_sq, state["n"])
        state["n"] += 1


def _mswag_particle(particle: Particle, dataloader, loss_fn: Callable,
                    pretrain_epochs: int, swag_epochs: int, swag_pids: list[int]) -> None:
    other_pids = [pid for pid in swag_pids if pid != particle.pid]
    
    # Pre-training loop
    for e in tqdm(range(pretrain_epochs)):
        losses = []
        for data, label in dataloader:
            fut = particle.step(loss_fn, data, label)
            futs = [particle.send(pid, "SWAG_STEP", loss_fn, data, label) for pid in other_pids]
            losses += [fut.wait()]
        print("Average epoch loss", torch.mean(torch.tensor(losses)))
    
    # Initialize swag
    [particle.send(pid, "SWAG_SWAG", True) for pid in other_pids]
    _swag_swag(particle, True)
    
    # Swag epochs
    for e in tqdm(range(swag_epochs)):
        losses = []
        for data, label in dataloader:
            # Update
            futs = [particle.send(pid, "SWAG_STEP", loss_fn, data, label) for pid in other_pids]
            fut = particle.step(loss_fn, data, label)
            [f.wait() for f in futs]
            losses += [fut.wait()]
        futs = [particle.send(pid, "SWAG_SWAG", False) for pid in other_pids]
        _swag_swag(particle, False)
        [f.wait() for f in futs]
        print("Average epoch loss", torch.mean(torch.tensor(losses)))


class MultiSWAG(Infer):
    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        super(MultiSWAG, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        self.swag_pids = []
        self.sample_pids = []
        
    def bayes_infer(self,
                    dataloader: DataLoader, 
                    loss_fn=torch.nn.MSELoss(),
                    num_models=1, lr=1e-3, pretrain_epochs=10, swag_epochs=5,
                    mswag_entry=_mswag_particle, mswag_state={}, f_save=False,
                    mswag_sample_entry=_swag_sample2, mswag_sample=_swag_sample2):
        if "n" in mswag_state:
            raise ValueError(f"Cannot run with state {mswag_state['n']}. Please rename.")
        mswag_state["n"] = 1

        def mk_swag(model_num):
            # Particle for parameter
            if model_num == 0:
                param_pid = self.push_dist.p_create(mk_optim, device=(model_num % self.num_devices), receive={
                    "SWAG_PARTICLE": mswag_entry,
                    "SWAG_SAMPLE_ENTRY": mswag_sample_entry,
                }, state=mswag_state)
            else:
                param_pid = self.push_dist.p_create(mk_optim, device=(model_num % self.num_devices), receive={
                    "SWAG_STEP": _swag_step,
                    "SWAG_SWAG": _swag_swag,
                    "SWAG_SAMPLE": mswag_sample
                }, state=mswag_state)
            return param_pid

        self.swag_pids = [mk_swag(n) for n in range(num_models)]
        for pid in self.swag_pids:
            if pid in mswag_state:
                raise ValueError(f"Cannot run with state {pid}. Please rename.")

        self.push_dist.p_wait([self.push_dist.p_launch(self.swag_pids[0], "SWAG_PARTICLE", dataloader, loss_fn, pretrain_epochs, swag_epochs, self.swag_pids)])

        if f_save:
            self.push_dist.save()

    def posterior_pred(self, dataloader: DataLoader, loss_fn=torch.nn.MSELoss(),
                       num_samples=20, scale=1.0, var_clamp=1e-30):
        # self.push_dist.p_wait([self.push_dist.p_launch(pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples) for pid in self.swag_pids])
        self.push_dist.p_wait([self.push_dist.p_launch(0, "SWAG_SAMPLE_ENTRY", dataloader, loss_fn, scale, var_clamp, num_samples, len(self.swag_pids))])


# =============================================================================
# Multi-Swag Training
# =============================================================================

def train_mswag(dataloader: DataLoader,
                loss_fn: Callable,
                pretrain_epochs: int,
                swag_epochs: int,
                num_models: int,
                cache_size: int,
                view_size:int,
                nn: Callable, *args,
                num_devices=1, lr=1e-3,
                mswag_entry=_mswag_particle, mswag_state={}, f_save=False,
                mswag_sample_entry=_swag_sample2,
                mswag_sample=_swag_sample2):
    mswag = MultiSWAG(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    mswag.bayes_infer(dataloader, loss_fn, num_models, lr=lr, pretrain_epochs=pretrain_epochs,
                      swag_epochs=swag_epochs, mswag_entry=mswag_entry, mswag_state=mswag_state,
                      f_save=f_save, mswag_sample_entry=mswag_sample_entry, mswag_sample=mswag_sample)
    return mswag
