
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import *

from push.bayes.infer import Infer
from push.particle import Particle
from push.lib.utils import detach_to_cpu
import torch.optim.lr_scheduler as lr_scheduler


# =============================================================================
# Helper
# =============================================================================
def create_optimizer(lr):
    """
    Create a function that returns Adam optimizer with a specific learning rate.
    
    Args:
        lr (float): Learning rate for the optimizer.
    
    Returns:
        function: Function that generates Adam optimizer with the specified learning rate.
    """
    def mk_optim(params):
        """
        Returns Adam optimizer with the specified learning rate.
        
        Args:
            params: Model parameters.
        
        Returns:
            torch.optim.Adam: Adam optimizer.
        """
        return torch.optim.Adam(params, lr=lr)
    
    return mk_optim

def mk_optim(params):
    """
    Returns Adam optimizer.
    
    Args:
        params: Model parameters.
    
    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    return torch.optim.Adam(params, lr=0.03)
    # return torch.optim.Adam(params, lr=1e-4)
    # return torch.optim.Adam(params, lr=1e-4, weight_decay=1e-2)

def mk_empty_scheduler(optim):
    """
    Returns Adam optimizer.
    
    Args:
        params: Model parameters.
    
    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    return None

def mk_scheduler(optim):
    """
    Returns Adam optimizer.
    
    Args:
        params: Model parameters.
    
    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    # return lr_scheduler.StepLR(optim, step_size=200, gamma=0.1)
    # return lr_scheduler.ExponentialLR(optim, gamma=0.1, last_epoch=-1, verbose='deprecated')
    return lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=1.0, total_iters=1)

# =============================================================================
# Deep Ensemble Training
# =============================================================================

def _deep_ensemble_main(particle: Particle, dataloader: DataLoader, loss_fn: Callable, epochs: int, bootstrap = True) -> None:
    """
    Main training loop for the lead particle in a deep ensemble.

    Args:
        particle (Particle): The lead particle to be trained in the deep ensemble.
        dataloader (DataLoader): The DataLoader containing training data.
        loss_fn (Callable): The loss function used for training.
        epochs (int): The number of training epochs.

    Returns:
        None

    Note:
        This function iteratively trains the lead particle for the specified number of epochs using the provided
        DataLoader and loss function. The lead particle also communicates with other particles in the ensemble during training,
        instructing them to step through the batch and training loop in a coordinated manner.
    """
    if bootstrap:
        other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
        num_ensembles = len(other_particles) + 1
        
        def generate_bootstrap(seed, n_samples):
            torch.manual_seed(seed)
            return torch.randint(0, n_samples, (n_samples,), dtype=torch.long)

        def build_bootstrap_datasets(n_estimators, data_loader):
            X, Y = [], []
            for inputs, labels in data_loader:
                X.append(inputs)
                Y.append(labels)
            X = torch.cat(X, dim=0)
            Y = torch.cat(Y, dim=0)
            
            n_samples = len(X)
            bootstraps = [generate_bootstrap(42 * i, n_samples) for i in range(1, n_estimators + 1)]
            dataloaders = []
            for indices in bootstraps:
                X_b = torch.index_select(X, 0, indices)
                Y_b = torch.index_select(Y, 0, indices)
                dataset = TensorDataset(X_b, Y_b)
                dataloader = DataLoader(dataset, batch_size=data_loader.batch_size, shuffle=True)
                dataloaders.append(dataloader)
            return dataloaders


        bootstrap_dataloaders = build_bootstrap_datasets(num_ensembles, dataloader)
        
        # Training loop
        tq = tqdm(range(epochs))
        for e in tq:
            losses = []
            for i, dataloader in enumerate(bootstrap_dataloaders):
                if i == 0:
                    for data, label in dataloader:
                        loss = particle.step(loss_fn, data, label).wait()
                        losses += [loss]
                else:
                    for data, label in dataloader:
                        particle.send(other_particles[i-1], "ENSEMBLE_STEP", loss_fn, data, label)

            tq.set_postfix({'loss': torch.mean(torch.tensor(losses))})
            # for data, label in dataloader:
            #     loss = particle.step(loss_fn, data, label).wait()
            #     losses += [loss]
            #     for pid in other_particles:
            #         particle.send(pid, "ENSEMBLE_STEP", loss_fn, data, label)
            # for pid in other_particles:
            #         particle.send(pid, "SCHEDULER_STEP")
            # tq.set_postfix({'loss': torch.mean(torch.tensor(losses))})

    else:
        other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
        # Training loop
        tq = tqdm(range(epochs))
        for e in tq:
            losses = []
            for data, label in dataloader:
                loss = particle.step(loss_fn, data, label).wait()
                losses += [loss]
                for pid in other_particles:
                    particle.send(pid, "ENSEMBLE_STEP", loss_fn, data, label)
            for pid in other_particles:
                    particle.send(pid, "SCHEDULER_STEP")
            tq.set_postfix({'loss': torch.mean(torch.tensor(losses))})
            # print(f"Average loss {particle.pid}", torch.mean(torch.tensor(losses)))
        # print(f"Average loss {particle.pid}", torch.mean(torch.tensor(losses)))


def _ensemble_step(particle: Particle, loss_fn: Callable, data, label, *args) -> None:
    """
    Perform a single step of ensemble training for a particle.

    Args:
        particle (Particle): The particle to perform the ensemble step.
        loss_fn (Callable): The loss function used for training.
        data: The input data for training.
        label: The labels corresponding to the input data.
        *args: Additional arguments for the ensemble step.

    Returns:
        None

    Note:
        This function performs a single step of ensemble training for the specified particle. It calls the
        particle's step method with the provided loss function, input data, and labels. Additional arguments
        can be passed for customization during training.
    """
    particle.step(loss_fn, data, label, *args)

def _ensemble_scheduler_step(particle: Particle, *args) -> None:
    """
    Perform a single step of ensemble training for a particle.

    Args:
        particle (Particle): The particle to perform the ensemble step.
        loss_fn (Callable): The loss function used for training.
        data: The input data for training.
        label: The labels corresponding to the input data.
        *args: Additional arguments for the ensemble step.

    Returns:
        None

    Note:
        This function performs a single step of ensemble training for the specified particle. It calls the
        particle's step method with the provided loss function, input data, and labels. Additional arguments
        can be passed for customization during training.
    """
    particle.scheduler_step(*args)


# =============================================================================
# Deep Ensemble Inference
# =============================================================================

def _leader_pred_dl(particle: Particle, dataloader: DataLoader, f_reg: bool = True, mode=["mean"]) -> dict:
    """
    Generate predictions using the lead particle in a deep ensemble for a DataLoader.

    Args:
        particle (Particle): The lead particle used for generating predictions.
        dataloader (DataLoader): The DataLoader containing input data for which predictions are to be generated.
        f_reg (bool, optional): Flag indicating whether this is a regression task. Set to false for classification tasks.
            Defaults to True. If set to True, the task is treated as a regression task; otherwise, it is treated as a classification task.
        mode (str, optional): The mode for generating predictions.
            Options include "mean" for mean predictions, "median" for median predictions,
            "min" for minimum predictions, and "max" for maximum predictions.
            Defaults to "mean".

    Returns:
        torch.Tensor: The ensemble predictions for the input data in the DataLoader.

    Note:
        This function generates predictions using the lead particle in a deep ensemble for each batch in the DataLoader.
        It internally calls the `_leader_pred` function to obtain predictions for each batch, and then concatenates
        the results into a single tensor. The `f_reg` flag determines whether the task is treated as a regression or classification task.
    """
    acc = []
    for data, label in dataloader:
        acc += [_leader_pred(particle, data, f_reg=f_reg, mode=mode)]
    results_dict = {}
    for mode_val in mode:
        results_dict[mode_val] = torch.cat([result[mode_val] for result in acc], dim=0)
    return results_dict


def _leader_pred(particle: Particle, data: torch.Tensor, f_reg: bool = True, mode=["mean"]) -> torch.Tensor:
    """
    Generate predictions using the lead particle in a deep ensemble.

    Args:
        particle (Particle): The lead particle used for generating predictions.
        data (torch.Tensor): The input data for which predictions are to be generated.
        f_reg (bool, optional): Flag indicating whether this is a regression task. Set to false for classification tasks.
        mode (str, optional): The mode for generating predictions.
            Options include "mean" for mean predictions, "median" for median predictions,
            "min" for minimum predictions, and "max" for maximum predictions.
            Defaults to "mean".

    Returns:
        torch.Tensor: The ensemble predictions for the input data.

    Raises:
        ValueError: If the specified mode is not supported.

    Note:
        This function generates predictions using the lead particle in a deep ensemble. It communicates with
        other particles in the ensemble to collect predictions, and then combines them based on the specified mode.
    """
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
    preds = []
    preds += [detach_to_cpu(particle.forward(data).wait())]
    for pid in other_particles:
        preds += [particle.send(pid, "ENSEMBLE_PRED", data).wait()]
    # print("preds: ", preds)
    t_preds = torch.stack(preds, dim=1)
    # print("t_preds: ", t_preds)
    # print("preds len:  ", len(preds))
    # print("preds: ", preds)
    # print("len preds[0]: ", len(preds[0]))
    # print("t_preds size: ", t_preds.size())
    results_dict = {}
    if f_reg:
        valid_modes = ["mean", "median", "min", "max", "std", "pred"]
        for mode_val in mode:
            assert mode_val in valid_modes, f"Mode {mode_val} not supported. Valid modes are {valid_modes}."
        if "std" in mode:
            results_dict["std"] = t_preds.std(dim=1)
        if "mean" in mode:
            results_dict["mean"] = t_preds.mean(dim=1)
        if "median" in mode:
            results_dict["median"] = t_preds.median(dim=1).values
        if "min" in mode:
            results_dict["min"] = t_preds.min(dim=1).values
        if "max" in mode:
            results_dict["max"] = t_preds.max(dim=1).values
        if "pred" in mode:
            results_dict["pred"] = t_preds
    else:
        valid_modes = ["logits", "mode", "std", "prob"]
        for mode_val in mode:
            assert mode_val in valid_modes, f"Mode {mode_val} not supported. Valid modes are {valid_modes}." 

        
        t_preds_softmax = [entry.softmax(dim=1) for entry in t_preds]
        stacked_preds = torch.stack(t_preds_softmax)
        if "logits" in mode:
            results_dict["logits"] = t_preds
        if "prob" in mode:
            results_dict["prob"] = stacked_preds
        if "mode" in mode:
            cls = [tensor_list.argmax(dim=1) for tensor_list in t_preds_softmax]
            stacked_cls = torch.stack(cls)
            results_dict["mode"] = torch.mode(stacked_cls, dim=1).values
        if "std" in mode:
            results_dict["std"] = torch.std(stacked_preds, dim=1)
    return results_dict


def _ensemble_pred(particle: Particle, data) -> None:
    """
    Generate ensemble predictions using the specified particle.

    Args:
        particle (Particle): The particle used for generating ensemble predictions.
        data: The input data for which predictions are to be generated.

    Returns:
        None

    Note:
        This function performs ensemble predictions using the provided particle. It calls the particle's forward
        method to generate predictions for the given input data. The predictions are then detached and transferred
        to the CPU for further processing.
    """
    return detach_to_cpu(particle.forward(data).wait())


# =============================================================================
# Deep Ensemble
# =============================================================================

class Ensemble(Infer):
    """The Ensemble Class.
    Used for running deep ensembles.

    Args:
        mk_nn (Callable): The base model to be ensembled.
        *args (any): Any arguments required for base model to be initialized.
        num_devices (int, optional): The desired number of gpu devices that will be utilized. Defaults to 1.
        cache_size (int, optional): The size of cache used to store particles. Defaults to 4.
        view_size (int, optional): The number of particles to consider storing in cache. Defaults to 4.
    """
    def __init__(self, mk_nn: Callable, *args: any, num_devices: int = 1, cache_size: int = 4, view_size: int = 4) -> None:
        super(Ensemble, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        
    def bayes_infer(self,
                    dataloader: DataLoader, epochs: int,
                    loss_fn: Callable = torch.nn.MSELoss(), lr: float = 0.01,
                    num_ensembles: int = 2, mk_scheduler = mk_scheduler,
                    prior = False, random_seed = False, bootstrap = False,
                    ensemble_entry=_deep_ensemble_main, ensemble_state={}, f_save: bool = False):
        """
        Creates particles and launches push distribution training loop.

        Args:
            dataloader (Callable): Dataloader.
            epochs (int, optional): Number of epochs to train for.
            loss_fn (Callable): Loss function to be used during training.
            num_ensembles (int, optional): The number of models to be ensembled.
            mk_optim (any): Returns an optimizer.
            ensemble_entry (function): Training loop for deep ensemble.
            ensemble_state (dict): A dictionary to store state variables for ensembled models.
                                   For example, in SWAG, we need to know how many SWAG epochs have passed
                                   to properly calculate a running average of model weights.
            f_save (bool): Flag to save each particle/model. Requires "particles" folder in the root directory
                           of the script calling train_deep_ensemble.

        Returns:
            None
        """
        mk_optim = create_optimizer(lr)
        if random_seed:
            train_keys = torch.randint(0, int(1e9), (num_ensembles,), dtype=torch.int64).tolist()
        else:
            train_keys = [-1] * num_ensembles
        # 1. Create particles
        pids = [
            self.push_dist.p_create(mk_optim, mk_scheduler, prior, train_keys[0], device=(0 % self.num_devices), receive={
                "ENSEMBLE_MAIN": ensemble_entry,
                "LEADER_PRED_DL": _leader_pred_dl,
                "LEADER_PRED": _leader_pred,
            }, state=ensemble_state)]
        for n in range(1, num_ensembles):
            pids += [self.push_dist.p_create(mk_optim, mk_scheduler, prior, train_keys[n], device=(n % self.num_devices), receive={
                "ENSEMBLE_STEP": _ensemble_step,
                "ENSEMBLE_PRED": _ensemble_pred,
                "SCHEDULER_STEP": _ensemble_scheduler_step
            }, state={})]
        # 2. Perform independent training
        self.push_dist.p_wait([self.push_dist.p_launch(0, "ENSEMBLE_MAIN", dataloader, loss_fn, epochs, bootstrap)])

        if f_save:
            self.push_dist.save()

    def posterior_pred(self, data: DataLoader, f_reg=True, mode = ["mean"]) -> torch.Tensor:
        """
        Generate posterior predictions for the given data.

        Args:
            data (Union[torch.Tensor, DataLoader]): The input data for which predictions are to be generated.
                If a torch.Tensor is provided, it is treated as a single input instance.
                If a DataLoader is provided, predictions are generated for all instances in the DataLoader.
            f_reg (bool, optional): Flag indicating whether this is a regression task. Set to false for classification tasks.
            mode (str, optional): The mode for generating predictions. Options include "mean" for mean predictions, "median"
                for median predictions, "max" for max predictions, and "min" for min predictions.
                Defaults to "mean".

        Returns:
            torch.Tensor: The posterior predictions for the input data.

        Raises:
            ValueError: If the provided data is not of type torch.Tensor or DataLoader.

        Note:
            This function uses the push_dist module to launch distributed predictions asynchronously.
            The type of predictions depends on the specified mode.
        """
        if isinstance(data, torch.Tensor):
            fut = self.push_dist.p_launch(0, "LEADER_PRED", data, f_reg, mode)
            return self.push_dist.p_wait([fut])[fut._fid]
        elif isinstance(data, DataLoader):
            fut = self.push_dist.p_launch(0, "LEADER_PRED_DL", data, f_reg, mode)
            return self.push_dist.p_wait([fut])[fut._fid]
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")


# =============================================================================
# Deep Ensemble Training
# =============================================================================

def train_deep_ensemble(dataloader: Callable, loss_fn: Callable, epochs: int, 
                        nn: Callable, *args, lr: float = 0.01, num_devices: int = 1, cache_size: int = 4, view_size: int = 4,
                        num_ensembles: int = 2, prior = False, random_seed = False, bootstrap = False,
                        ensemble_entry = _deep_ensemble_main, ensemble_state={}) -> List[torch.Tensor]:
    """Train a deep ensemble PusH distribution and return a list of particle parameters.

    Args:
        dataloader (Callable): Dataloader.
        loss_fn (Callable): Loss function to be used during training.
        epochs (int, optional): Number of epochs to train for.
        nn (Callable): The base model to be ensembled and trained.
        *args (any): Any arguments needed for the model's initialization.
        num_devices (int, optional): The desired number of gpu devices to be utilized during training. Defaults to 1.
        cache_size (int, optional): The desired size of cache allocated to storing particles. Defaults to 4.
        view_size (int, optional): The number of other particle's parameters that can be seen by a particle on a single GPU. Defaults to 4.
        num_ensembles (int, optional): The number of models to be ensembled. Defaults to 2.
        mk_optim (any, optional): Returns an optimizer. Defaults to mk_optim.
        ensemble_entry (function, optional): Training loop for deep ensemble. Defaults to _deep_ensemble_main.
        ensemble_state (dict, optional): a dictionary to store state variables for ensembled models. i.e. in swag we need to know how
           how many swag epochs have passed to properly calculate a running average of model weights. Defaults to {}.
    Returns:
        List[torch.Tensor]: Returns a list of all particle's parameters.
    """
    ensemble = Ensemble(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
    ensemble.bayes_infer(dataloader, epochs, loss_fn=loss_fn, lr=lr, num_ensembles=num_ensembles,
                         ensemble_entry=ensemble_entry, ensemble_state=ensemble_state, prior=prior, bootstrap=bootstrap, random_seed=random_seed)
    return ensemble
