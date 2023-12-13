from torch.utils.data import DataLoader
from typing import *
import torch
from push.bayes.infer import Infer
from push.particle import Particle
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu
from tqdm import tqdm


# =============================================================================
# Helper functions
# =============================================================================

def mk_optim(params):
    """Returns an Adam optimizer.

    Args:
        params: Parameters for optimization.

    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    return torch.optim.Adam(params, lr=1e-3, weight_decay=1e-2)


# =============================================================================
# Swag Training
# =============================================================================

def _swag_step(particle: Particle,
               loss_fn: Callable,
               data: torch.Tensor,
               label: torch.Tensor,
               *args: Any) -> None:
    """Calls one SWAG particle's step function.

    Args:
        particle (Particle): SWAG particle.
        loss_fn (Callable): Loss function.
        data (torch.Tensor): Input data.
        label (torch.Tensor): Ground truth labels.
        *args: Additional arguments.
    """
    particle.step(loss_fn, data, label, *args)


def update_theta(state, state_sq, param, param_sq, n):
    """Updates the first and second moments and iterates the number of parameter settings averaged.

    Args:
        state: First moment.
        state_sq: Second moment.
        param: Parameters.
        param_sq: Squared parameters.
        n (int): Number of iterations.
    """
    for st, st_sq, p, p_sq in zip(state, state_sq, param, param_sq):
        st.data = (st.data * n + p.data)/(n+1)
        st_sq.data = (st_sq.data * n + p_sq.data)/(n+1)


def _swag_swag(particle: Particle, reset: bool) -> None:
    """Initializes or updates moments for SWAG.

    Args:
        particle (Particle): SWAG particle.
        reset (bool): Whether to reset or update moments.
    """
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
    """Training function for MSWAG particle.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        pretrain_epochs (int): Number of pre-training epochs.
        swag_epochs (int): Number of SWAG epochs.
        swag_pids (list[int]): List of SWAG particle IDs.
    """
    other_pids = [pid for pid in swag_pids if pid != particle.pid]
    
    # Pre-training loop
    for e in tqdm(range(pretrain_epochs)):
        losses = []
        for data, label in dataloader:
            fut = particle.step(loss_fn, data, label)
            futs = [particle.send(pid, "SWAG_STEP", loss_fn, data, label) for pid in other_pids]
            losses += [fut.wait()]
        # print("Average epoch loss", torch.mean(torch.tensor(losses)))
    
    # Initialize SWAG
    [particle.send(pid, "SWAG_SWAG", True) for pid in other_pids]
    _swag_swag(particle, True)
    
    # SWAG epochs
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
        # print("Average epoch loss", torch.mean(torch.tensor(losses)))


# =============================================================================
# SWAG Inference
# =============================================================================       

def _leader_pred(particle: Particle,
                        dataloader: DataLoader, scale: float,
                        var_clamp: float, num_samples: int,
                        mode: List[str], num_models: int, f_reg: bool = True) -> dict:
    """Generate MSWAG predictions using the lead particle in a MSWAG PusH distribution.

    Args:
        particle (Particle): The lead particle in the MSWAG ensemble.
        dataloader (DataLoader): DataLoader for input data.
        scale (float): Scaling factor for MSWAG sampling.
        var_clamp (float): Clamping value for variance in MSWAG sampling.
        num_samples (int): Number of SWAG samples.
        mode (str): Ensemble prediction mode. Options: "mode", "mean".
        num_models (int): Number of models in the ensemble.

    Returns:
        results_dict (dict): Ensemble predictions for each mode. Access
    """
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
    preds = []
     # Perform MSWAG sampling for the lead particle
    preds += [_mswag_pred(particle, dataloader, var_clamp, scale, num_samples)]

    for pid in other_particles:
        preds += [particle.send(pid, "SWAG_PRED", dataloader, scale, var_clamp, num_samples).wait()]
    t_preds = [torch.cat(tensor_list, dim=0)for tensor_list in preds]
    results_dict = {}
    if f_reg:
        valid_modes = ["mean", "median", "min", "max", "std"]
        for mode_val in mode:
            assert mode_val in valid_modes, f"Mode {mode_val} not supported. Valid modes are {valid_modes}."
        stacked_preds = torch.stack(t_preds, dim=0)
        if "mean" in mode:
            results_dict["mean"] = torch.mean(stacked_preds, dim=0)
        if "median" in mode:
            results_dict["median"] = torch.median(stacked_preds, dim=0).values
        if "min" in mode:
            results_dict["min"] = torch.min(stacked_preds, dim=0).values
        if "max" in mode:
            results_dict["max"] = torch.max(stacked_preds, dim=0).values
        if "std" in mode:
            results_dict["std"] = torch.std(stacked_preds, dim=0)
    else:
        valid_modes = ["mode", "mean", "median"]
        for mode_val in mode:
            assert mode_val in valid_modes, f"Mode {mode_val} not supported. Valid modes are {valid_modes}."
        preds_softmax = [entry.softmax(dim=1) for entry in t_preds]
        if "mode" in mode:
            cls = [tensor_list.argmax(dim=1) for tensor_list in preds_softmax]
            stacked_cls = torch.stack(cls)
            results_dict["mode"] = torch.mode(stacked_cls, dim=0).values
        if "mean" in mode:
            stacked_preds = torch.stack(preds_softmax)
            mean_values = torch.mean(stacked_preds, dim=0)
            results_dict["mean"] = mean_values.argmax(dim=1)
        if "median" in mode:
            stacked_preds = torch.stack(preds_softmax)
            median_values = torch.median(stacked_preds, dim=0).values
            results_dict["median"] = median_values.argmax(dim=1)
    return results_dict


def _mswag_pred(particle: Particle,
                  dataloader: DataLoader,
                  scale: float,
                  var_clamp: float,
                  num_samples: int) -> torch.Tensor:
    """MSWAG sample prediction function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of SWAG samples.

    Returns:
        torch.Tensor: Predictions for the input data.
    """
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale ** 0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute and store prediction for each SWAG sample
    # preds = {i: [] for i in range(num_samples)}
    preds = []
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

        # Forward pass and store predictions
        pred = [detach_to_cpu(particle.forward(data).wait()) for data, _ in dataloader]
        preds += [pred]

    mean_preds = []
    # Calculate mean predictions over num_samples
    for n in range(len(preds[0])):
        # Accumulate predictions for each sample
        mean_preds_accum = sum(preds[i][n] for i in range(num_samples))
        # Calculate mean for each data point
        mean_preds.append(mean_preds_accum / num_samples)

    return mean_preds


def _mswag_sample_entry(particle: Particle,
                        dataloader: DataLoader,
                        loss_fn: Callable,
                        scale: float,
                        var_clamp: float,
                        num_samples: int,
                        num_models) -> None:
    """SWAG sampling entry function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
        num_models (int): Number of models.
    """
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
    """SWAG sampling function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
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
    """Regression version of SWAG sample entry function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
        num_models (int): Number of models.
    """
    # Unpack state
    state = particle.state
    pid = particle.pid
    other_pids = list(range(1, num_models))

    # Perform SWAG sampling for every particle
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [particle.send(pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples) for pid in other_pids]
    ans = [f.wait() for f in futs]

    # Mean prediction across all models
    mean_preds = torch.mean(torch.stack([my_ans['mean_preds']] + [ans[i-1]['mean_preds'] for i in range(1, num_models)]), dim=0)

    return mean_preds


def _mswag_sample_regression(particle: Particle,
                             dataloader: DataLoader,
                             loss_fn: Callable,
                             scale: float,
                             var_clamp: float,
                             num_samples: int) -> None:
    """
    Modified SWAG sample function for regression problems.

    Args:
        particle (Particle): The SWAG particle.
        dataloader (DataLoader): DataLoader containing the data.
        loss_fn (Callable): Loss function used for computing losses.
        scale (float): Scaling factor for the SWAG sample.
        var_clamp (float): Clamping value for the variance.
        num_samples (int): Number of SWAG samples to generate.

    Returns:
        dict: Dictionary containing computed losses and predictions.

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
    MultiSWAG class for running MultiSWAG models.

    Args:
        mk_nn (Callable): The base model to be ensembled.
        *args: Any arguments required for the base model initialization.
        num_devices (int): The desired number of GPU devices to utilize.
        cache_size (int): The size of the cache used to store particles.
        view_size (int): The number of particles to consider storing in the cache.

    """
    def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
        """
        Initialize the MultiSWAG model.

        Args:
            mk_nn (Callable): The base model to be ensembled.
            *args: Any arguments required for the base model initialization.
            num_devices (int): The desired number of GPU devices to utilize.
            cache_size (int): The size of the cache used to store particles.
            view_size (int): The number of particles to consider storing in the cache.

        """
        super(MultiSWAG, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        self.swag_pids = []
        self.sample_pids = []
        
    def bayes_infer(self,
                    dataloader: DataLoader, 
                    loss_fn=torch.nn.MSELoss(),
                    num_models=1, pretrain_epochs=10, swag_epochs=5,
                    mswag_entry=_mswag_particle, mswag_state={}, f_save=False,
                    mswag_sample_entry=_mswag_sample_entry, mswag_sample=_mswag_sample):
        """
        Perform Bayesian inference using MultiSWAG.

        Args:
            dataloader (DataLoader): DataLoader containing the data.
            loss_fn (Callable): Loss function used for training.
            num_models (int): Number of models to be ensembled.
            lr (float): Learning rate for training.
            pretrain_epochs (int): Number of epochs for pretraining.
            swag_epochs (int): Number of epochs for SWAG training.
            mswag_entry (Callable): Training loop for deep ensemble.
            mswag_state (dict): State variables for ensembled models.
            f_save (bool): Flag to save each particle/model.
            mswag_sample_entry (Callable): Sampling function.
            mswag_sample (Callable): MultiSWAG sample function.

        Returns:
            None

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
                    "LEADER_PRED": _leader_pred,
                }, state=mswag_state)
            else:
                param_pid = self.push_dist.p_create(mk_optim, device=(model_num % self.num_devices), receive={
                    "SWAG_STEP": _swag_step,
                    "SWAG_SWAG": _swag_swag,
                    "SWAG_SAMPLE": mswag_sample,
                    "SWAG_PRED": _mswag_pred
                }, state=mswag_state)
            return param_pid

        self.swag_pids = [mk_swag(n) for n in range(num_models)]
        for pid in self.swag_pids:
            if pid in mswag_state:
                raise ValueError(f"Cannot run with state {pid}. Please rename.")

        self.push_dist.p_wait([self.push_dist.p_launch(self.swag_pids[0], "SWAG_PARTICLE", dataloader, loss_fn, pretrain_epochs, swag_epochs, self.swag_pids)])

        if f_save:
            self.push_dist.save()

    def posterior_pred(self, data: DataLoader, loss_fn=torch.nn.MSELoss(),
                       num_samples: int = 20, scale: float = 1.0, var_clamp: float = 1e-30, mode: List[str] = ["mean"], f_reg: bool = True):
        """
        Generate posterior predictions using MultiSWAG.

        Args:
            dataloader (DataLoader): DataLoader containing the data.
            loss_fn (Callable): Loss function used for computing losses.
            num_samples (int): Number of samples to generate.
            scale (float): Scaling factor for the SWAG sample.
            var_clamp (float): Clamping value for the variance.

        Returns:
            None

        """

        # if isinstance(data, torch.Tensor):
        #     fut = self.push_dist.p_launch(0, "LEADER_PRED", data, scale, var_clamp, num_samples, mode, len(self.swag_pids))
        #     return self.push_dist.p_wait([fut])[fut._fid]
        if isinstance(data, DataLoader):
            fut = self.push_dist.p_launch(0, "LEADER_PRED", data, scale, var_clamp, num_samples, mode, len(self.swag_pids), f_reg)
            return self.push_dist.p_wait([fut])[fut._fid]
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")

# =============================================================================
# Multi-Swag Training
# =============================================================================

def train_mswag(dataloader: DataLoader, loss_fn: Callable, pretrain_epochs: int,
                swag_epochs: int, nn: Callable, *args, num_devices=1, cache_size: int = 4, view_size: int = 4,
                num_models: int, mswag_entry=_mswag_particle, mswag_state={}, f_save=False,
                mswag_sample_entry=_mswag_sample_entry, mswag_sample=_mswag_sample):
    """
    Train a MultiSWAG model.

    Args:
        dataloader (DataLoader): DataLoader containing the training data.
        loss_fn (Callable): Loss function used for training.
        pretrain_epochs (int): Number of epochs for pretraining.
        swag_epochs (int): Number of epochs for SWAG training.
        num_models (int): Number of models to use in MultiSWAG.
        cache_size (int): Size of the cache for MultiSWAG.
        view_size (int): Size of the view for MultiSWAG.
        nn (Callable): Callable function representing the neural network model.
        *args: Additional arguments for the neural network.
        num_devices (int): Number of devices for training (default is 1).
        lr (float): Learning rate for training (default is 1e-3).
        mswag_entry (Callable): MultiSWAG entry function (default is _mswag_particle).
        mswag_state (dict): Initial state for MultiSWAG (default is {}).
        f_save (bool): Flag to save the model (default is False).
        mswag_sample_entry (Callable): MultiSWAG sample entry function (default is _mswag_sample_entry).
        mswag_sample (Callable): MultiSWAG sample function (default is _mswag_sample).

    Returns:
        MultiSWAG: Trained MultiSWAG model.

    """
    mswag = MultiSWAG(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    mswag.bayes_infer(dataloader, loss_fn, num_models, pretrain_epochs=pretrain_epochs,
                      swag_epochs=swag_epochs, mswag_entry=mswag_entry, mswag_state=mswag_state,
                      f_save=f_save, mswag_sample_entry=mswag_sample_entry, mswag_sample=mswag_sample)
    return mswag
