from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
import torch

from push.bayes.infer import Infer
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu


# =============================================================================
# Helper
# =============================================================================

def mk_optim(params):
    """returns Adam optimizer"""
    return torch.optim.Adam(params, lr=1e-5, weight_decay=1e-3)


# =============================================================================
# Swag Training
# =============================================================================

def _swag_step(particle: Particle,
               loss_fn: Callable,
               data: torch.Tensor,
               label: torch.Tensor,
               *args: any) -> None:
    # calls one swag particle's step function
    particle.step(loss_fn, data, label,*args)


def update_theta(state, state_sq, param, param_sq, n):
    """

    Updates the first and second moments and iterates the number of parameter settings averaged.

    """
    for st, st_sq, p, p_sq in zip(state, state_sq, param, param_sq):
        st.data = (st.data * n + p.data)/(n+1)
        st_sq.data = (st_sq.data * n + p_sq.data)/(n+1)


def _swag_swag(particle: Particle, reset: bool) -> None:    
    # if reset, initializes mom1 and mom2, else updates mom1 and mom2
    state = particle.state
    if reset:
        state[particle.pid] = {
            "mom1": [param for param in particle.module.parameters()],
            "mom2": [param*param for param in particle.module.parameters()]
        }
    else:
        params = [param for param in particle.module.parameters()]
        params_sq = [param*param for param in particle.module.parameters()]
        update_theta(state[particle.pid]["mom1"], state[particle.pid]["mom2"], params, params_sq, state["n"])
        state["n"] += 1


def _mswag_particle(particle: Particle, dataloader, loss_fn: Callable,
                    pretrain_epochs: int, swag_epochs: int, swag_pids: list[int]) -> None:
    # training function for mswag particle
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


# =============================================================================
# SWAG Inference
# =============================================================================

def _mswag_sample_entry(particle: Particle,
                        dataloader: DataLoader,
                        loss_fn: Callable,
                        scale: float,
                        var_clamp: float,
                        num_samples: int,
                        num_models) -> None:
    # TODO comment explaining this function
    # Unpack state
    state = particle.state
    pid = particle.pid
    other_pids = list(range(1, num_models))

    # Perform SWAG sampling for every particle
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [particle.send(pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples) for pid in other_pids]
    ans = [f.wait() for f in futs]
    
    # Majority vote of majority prediction
    classes = {k: [0 for i in range(10)] for k in range(10)}
    max_preds = [my_ans['max_preds']] + [ans[i-1]['max_preds'] for i in range(1, num_models)]
    for idx, (data, label) in enumerate(dataloader):
        max_pred = torch.mode(torch.stack([max_preds[i][idx] for i in range(num_models)]), dim=0).values
        for x, y in zip(max_pred, label):
            classes[y.item()][x.item()] += 1
    
    # Majority vote across all SWAG samples
    all_preds = [my_ans['preds']] + [ans[i-1]['preds'] for i in range(1, num_models)]
    classes2 = {k: [0 for i in range(10)] for k in range(10)}
    for idx, (data, label) in enumerate(dataloader):
        preds = []
        for m in range(num_models):
            preds += [torch.stack([all_preds[m][i][idx] for i in range(num_samples)])]
        max_pred = torch.mode(torch.cat(preds, dim=0), dim=0).values
        for x, y in zip(max_pred, label):
            classes2[y.item()][x.item()] += 1

    return classes, classes2

def _mswag_sample(particle: Particle,
                  dataloader: DataLoader,
                  loss_fn: Callable,
                  scale: float,
                  var_clamp: float,
                  num_samples: int) -> None:
    """Inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    """
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    classes = {k: [0 for i in range(10)] for k in range(10)}
    
    losses = []
    for data, label in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, label)
        cls = pred.softmax(dim=1).argmax(dim=1)
        for x, y in zip(cls, label):
            classes[y.item()][x.item()] += 1
        losses += [loss.detach().to("cpu")]

    # Compute and store prediction for each SWAG sample
    preds = {i: [] for i in range(num_samples)}
    swag_losses = {}
    for i in range(num_samples):
        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        swag_losses = []
        for data, label in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            cls = pred.softmax(dim=1).argmax(dim=1)
            preds[i] += [cls]
            loss = loss_fn(pred, label)
            swag_losses += [loss.detach().to("cpu")]    
    
    # Majority prediction for the current SWAG particle
    max_preds = []
    for n in range(len(preds[0])):
        max_preds += [torch.mode(torch.stack([preds[i][n] for i in range(num_samples)]), dim=0).values]

    return {
        "classes": classes, 
        "losses": torch.mean(torch.tensor(losses)),
        "preds": preds,
        "max_preds": max_preds,
    }



def _mswag_sample_entry_regression(particle: Particle,
                                   dataloader: DataLoader,
                                   loss_fn: Callable,
                                   scale: float,
                                   var_clamp: float,
                                   num_samples: int,
                                   num_models) -> None:
    # Unpack state
    state = particle.state
    pid = particle.pid
    other_pids = list(range(1, num_models))

    # Perform SWAG sampling for every particle
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [particle.send(pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples) for pid in other_pids]
    ans = [f.wait() for f in futs]

    # Mean prediction across all models
    print("my_ans: ", my_ans)
    mean_preds = torch.mean(torch.stack([my_ans['mean_preds']] + [ans[i-1]['mean_preds'] for i in range(1, num_models)]), dim=0)

    return mean_preds


def _mswag_sample_regression(particle: Particle,
                             dataloader: DataLoader,
                             loss_fn: Callable,
                             scale: float,
                             var_clamp: float,
                             num_samples: int) -> None:
    """Modified for regression problems."""
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    losses = []
    for data, target in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, target)
        losses += [loss.detach().to("cpu")]

    # Compute and store prediction for each SWAG sample
    preds = {i: [] for i in range(num_samples)}
    swag_losses = {}
    for i in range(num_samples):
        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        swag_losses = []
        for data, target in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            preds[i] += [pred]
            loss = loss_fn(pred, target)
            swag_losses += [loss.detach().to("cpu")]

    # Mean prediction for the current SWAG particle
    mean_preds = torch.mean(torch.stack([torch.stack(v) for v in preds.values()]), dim=0)


    return {
        "losses": torch.mean(torch.tensor(losses)),
        "mean_preds": mean_preds,
        "swag_losses": swag_losses,
    }


class MultiSWAG(Infer):
    """

    MultiSWAG class.
    Used for running MultiSWAG models.

    :param Callable mk_nn: The base model to be ensembled.
    :param any *args: Any arguments required for base model to be initialized.
    :param int num_devices: The desired number of gpu devices that will be utilized.
    :param int cache_size: The size of cache used to store particles.
    :param int view_size: The number of particles to consider storing in cache.
    
    """
    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        super(MultiSWAG, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        self.swag_pids = []
        self.sample_pids = []
        
    def bayes_infer(self,
                    dataloader: DataLoader, 
                    loss_fn=torch.nn.MSELoss(),
                    num_models=1, lr=1e-3, pretrain_epochs=10, swag_epochs=5,
                    mswag_entry=_mswag_particle, mswag_state={}, f_save=False,
                    mswag_sample_entry=_mswag_sample_entry, mswag_sample=_mswag_sample):
        """
        Creates swag particles and launches a PusH distribution with MultiSWAG.
        
        :param Callable dataloader: Dataloader.
        :param int, optional epochs: Number of epochs to train for. 
        :param Callable loss_fn: Loss function to be used during training.
        :param int, optional num_ensembles: The number of models to be ensembled.
        :param any mk_optim: Returns an optimizer.
        :param mswag_entry: Training loop for deep ensemble.
        :param dict mswag_state: A dictionary to store state variables for ensembled models. i.e. in swag we need to know how
           how many swag epochs have passed to properly calculate a running average of model weights.
        :param mswag_sample_entry: Sampling function.
        :param bool f_save: Flag to save each particle/model. Requires "particles" folder in root directory of the script calling train_deep_ensemble

        :return: None

        """
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
                mswag_sample_entry=_mswag_sample_entry,
                mswag_sample=_mswag_sample):
    """
    Train a MultiSWAG model.

    :param DataLoader dataloader: DataLoader containing the training data.

    :param Callable loss_fn: Loss function used for training.

    :param int pretrain_epochs: Number of epochs for pretraining.

    :param int swag_epochs: Number of epochs for SWAG training.

    :param int num_models: Number of models to use in MultiSWAG.

    :param int cache_size: Size of the cache for MultiSWAG.

    :param int view_size: Size of the view for MultiSWAG.

    :param Callable nn: Callable function representing the neural network model.

    :param *args: Additional arguments for the neural network.

    :param int num_devices: Number of devices for training (default is 1).

    :param float lr: Learning rate for training (default is 1e-3).

    :param Callable mswag_entry: MultiSWAG entry function (default is _mswag_particle).

    :param dict mswag_state: Initial state for MultiSWAG (default is {}).

    :param bool f_save: Flag to save the model (default is False).

    :param Callable mswag_sample_entry: MultiSWAG sample entry function (default is _mswag_sample_entry).

    :param Callable mswag_sample: MultiSWAG sample function (default is _mswag_sample).

    :return MultiSWAG: Trained MultiSWAG model.
    """
    mswag = MultiSWAG(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    mswag.bayes_infer(dataloader, loss_fn, num_models, lr=lr, pretrain_epochs=pretrain_epochs,
                      swag_epochs=swag_epochs, mswag_entry=mswag_entry, mswag_state=mswag_state,
                      f_save=f_save, mswag_sample_entry=mswag_sample_entry, mswag_sample=mswag_sample)
    return mswag
