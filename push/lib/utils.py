import torch
from typing import *


def to_device(device: int, val: Union[Dict, List, Tuple, torch.Tensor]) -> Union[Dict, List, torch.Tensor]:
    if isinstance(val, dict):
        return {k: to_device(device, v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_device(device, v) for v in val]
    elif isinstance(val, tuple):
        return tuple(to_device(device, v) for v in val)
    elif isinstance(val, torch.Tensor):
        return val.to(device)
    else:
        return val


def detach_to_device(device: int, val: Union[Dict, List, Tuple, torch.Tensor], requires_grad=False) -> Union[Dict, List, torch.Tensor]:
    if isinstance(val, dict):
        return {k: detach_to_device(device, v, requires_grad=requires_grad) for k, v in val.items()}
    elif isinstance(val, list):
        return [detach_to_device(device, v, requires_grad=requires_grad) for v in val]
    elif isinstance(val, tuple):
        return tuple(detach_to_device(device, v, requires_grad=requires_grad) for v in val)
    elif isinstance(val, torch.Tensor):
        return val.detach().requires_grad_().to(device) if requires_grad and isinstance(val, torch.FloatTensor) else val.detach().to(device)
    else:
        return val


def detach_to_cpu(val: Union[Dict, List, torch.Tensor]) -> Union[Dict, List, torch.Tensor]:
    if isinstance(val, dict):
        return {k: detach_to_cpu(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [detach_to_cpu(v) for v in val]
    elif isinstance(val, torch.Tensor):
        return val.detach().to("cpu")
    else:
        return val
