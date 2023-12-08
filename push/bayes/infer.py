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

    def get_models(self):
        """Instantiates models with particle parameters.

        Returns:
            List[torch.nn.Module]: List of instantiated models with particle parameters.

        Raises:
            AssertionError: If the size of the parameters for a model does not match the size of the model's parameters.
        """ 
        models = []
        p_parameters = self.p_parameters()
        for i in range(len(p_parameters)):
            model = self.mk_nn(*self.args)
            assert len(p_parameters[i]) == len(list(model.parameters())), "Model {i}'s parameter size must equal particle {i}'s"
            for model_param, param_value in zip(model.parameters(), p_parameters[i]):
                model_param.data = param_value
            models.append(model)
        return models

    def get_output(self, dataloader: DataLoader) -> List[List[torch.Tensor]]:
        """Calculates model predictions on a given dataloader.

        Args:
            dataloader (DataLoader): DataLoader containing the input data.

        Returns:
            List[List[torch.Tensor]]: List of model predictions for each batch.

        """
        # Instanstiate models
        models = self.get_models()
        outputs = []
    
        for i in range(len(models)):
            model_i_output = []
            # Calculate output for each model
            for data, label in dataloader:
                model_i_output.extend(models[i](data).detach())
            outputs.append(model_i_output)
        return outputs

    def get_avg_pred(self, outputs: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Calculates the average predictions over different models.

        Args:
            outputs (List[List[torch.Tensor]]): List of model predictions for each batch.

        Returns:
            List[torch.Tensor]: List of tensors representing the average predictions over different models.

        """
        transposed_lists = list(map(list, zip(*outputs)))
        averages = []

        # Calculate the average predictions over each model
        for tensors in transposed_lists:
            stacked_tensor = torch.stack(tensors)
            average_tensor = torch.mean(stacked_tensor, dim=0)
            averages.append(average_tensor)
        return averages

    def get_median_pred(self, outputs: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Calculates the median predictions over different models.

        Args:
            outputs (List[List[torch.Tensor]]): List of model predictions for each batch.

        Returns:
            List[torch.Tensor]: List of tensors representing the median predictions over different models.

        """
        transposed_lists = list(map(list, zip(*outputs)))
        medians = []

        # Calculate the median predictions over each model
        for tensors in transposed_lists:
            stacked_tensor = torch.stack(tensors)
            median_tensor = torch.median(stacked_tensor, dim=0).values
            medians.append(median_tensor)
        return medians

    def get_var(self, outputs: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Calculates the variance of predictions over different models.

        Args:
            outputs (List[List[torch.Tensor]]): List of model predictions for each batch.

        Returns:
            List[torch.Tensor]: List of tensors representing the variance of predictions over different models.

        """
        transposed_lists = list(map(list, zip(*outputs)))
        variances = []

        # Calculate the variance for each list of tensors and add them to the 'variances' list
        for tensors in transposed_lists:
            stacked_tensor = torch.stack(tensors)
            variance_tensor = torch.var(stacked_tensor, dim=0)
            variances.append(variance_tensor)
        return variances

    def _cleanup(self):
        self.push_dist._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.push_dist.__exit__(exc_type, exc_value, traceback)