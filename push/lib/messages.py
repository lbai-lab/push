import torch
from typing import *


class MSG:
    """
    Base class for messages in the system.
    """
    pass


# =============================================================================
# Node Event Loop Messages
# =============================================================================

class NodeEvtLoopInitMSG(MSG):
    """
    Message indicating the initialization of the Node Event Loop.
    """
    def __init__(self):
        pass


class NodeEvtLoopCleanupMSG(MSG):
    """
    Message indicating the cleanup of the Node Event Loop.
    """
    def __init__(self):
        pass


class NELBroadcastParticlesMSG(MSG):
    """
    Message for broadcasting particles in the Node Event Loop.

    Attributes:
        in_queues (List[Queue]): List of input queues.
        out_queues (List[Queue]): List of output queues.
        particle_to_device (Dict[int, int]): Mapping of particle ids to devices.
    """
    def __init__(self, in_queues: List[Any], out_queues: List[Any], particle_to_device: Dict[int, int]):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.particle_to_device = particle_to_device

    def __str__(self) -> str:
        return f"NELBroadcastParticlesMSG({self.in_queues}, {self.out_queues}, {self.particle_to_device})"


class NELBroadcastParticlesAckMSG(MSG):
    """
    Acknowledgment message for broadcasting particles in the Node Event Loop.
    """
    def __init__(self):
        pass


class NELSaveModel(MSG):
    """
    Message for saving the model in the Node Event Loop.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and file id.
    """
    def __init__(self, pid_fid: Tuple[int, int]):
        self.pid_fid = pid_fid

    def __str__(self) -> str:
        return f"NELSaveModel({self.pid_fid})"


class NELSaveModelAckPDMSG(MSG):
    """
    Acknowledgment message for saving the model in the Node Event Loop.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and file id.
    """
    def __init__(self, pid_fid: Tuple[int, int]):
        self.pid_fid = pid_fid

    def __str__(self) -> str:
        return f"NELSaveModelAckPDMSG({self.pid_fid})"


# =============================================================================
# Push Distribution Messages
# =============================================================================

# -----------------------------------------------------
# One-Time
# -----------------------------------------------------

class ReceiveParticleInitPDMSG(MSG):
    """
    Message for initializing particle reception in Push Distribution.

    Attributes:
        device (int): The device id.
        pid (int): The particle id.
        mk_optim (Callable): The function to create a new optimizer.
        receive (Callable): The function to receive particles.
        state (Any): State information for initialization.
    """
    def __init__(self, device: int, pid: int, mk_optim: Callable, mk_scheduler: Callable, receive: Callable, state: Any):
        self.device = device
        self.pid = pid
        self.mk_optim = mk_optim
        self.mk_scheduler = mk_scheduler
        self.receive = receive
        self.state = state

    def __str__(self) -> str:
        return f"ReceiveParticleInitPDMSG({self.device}, {self.pid}, {self.mk_optim}, {self.mk_scheduler} ,{self.receive}, {self.state})"


class ReceiveParticleInitAckPDMSG(MSG):
    """
    Acknowledgment message for particle initialization in Push Distribution.
    """
    def __init__(self):
        pass


class ReceiveRegisterPDMSG(MSG):
    """
    Message for registering particle reception in Push Distribution.

    Attributes:
        pid (int): The particle id.
        msg (str): The message identifier.
        fn (Callable): The function associated with the registration.
        state (Dict[str, Any]): State information for registration.
    """
    def __init__(self, pid: int, msg: str, fn: Callable, state: Dict[str, Any]):
        self.pid = pid
        self.msg = msg
        self.fn = fn
        self.state = state

    def __str__(self) -> str:
        return f"ReceiveRegisterPDMSG({self.pid}, {self.msg}, {self.fn}, {self.state})"


class ReceiveRegisterAckPDMSG(MSG):
    """
    Acknowledgment message for registering particle reception in Push Distribution.
    """
    def __init__(self):
        pass


# -----------------------------------------------------
# Multi-Time
# -----------------------------------------------------

class ReceiveFuncPDMSG(MSG):
    """
    Message for receiving function calls in Push Distribution.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        pid_to (int): The target particle id.
        msg (str): The message identifier.
        args (List[Any]): List of arguments for the function call.
    """
    def __init__(self, pid_fid: Tuple[int, int], pid_to: int, msg: str, args: List[Any]):
        self.pid_fid = pid_fid
        self.pid_to = pid_to
        self.msg = msg
        self.args = args

    def __str__(self) -> str:
        return f"ReceiveFuncPDMSG({self.pid_fid}, {self.pid_to}, {self.msg}, {self.args})"


class ReceiveFuncAckPDMSG(MSG):
    """
    Acknowledgment message for receiving function calls in Push Distribution.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        result (Any): The result of the function call.
    """
    def __init__(self, pid_fid: Tuple[int, int], result: Any):
        self.pid_fid = pid_fid
        self.result = result

    def __str__(self) -> str:
        return f"ReceiveFuncAckPDMSG({self.pid_fid}, {self.result})"


class ReceiveParametersPDMSG(MSG):
    """
    Message for receiving parameters in Push Distribution.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        pid (int): The particle id.
    """
    def __init__(self, pid_fid: Tuple[int, int], pid: int):
        self.pid_fid = pid_fid
        self.pid = pid

    def __str__(self) -> str:
        return f"ReceiveParametersPDMSG({self.pid_fid}, {self.pid})"


class ReceiveParametersAckPDMSG(MSG):
    """
    Acknowledgment message for receiving parameters in Push Distribution.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        params (List[torch.Tensor]): List of parameters.
    """
    def __init__(self, pid_fid: Tuple[int, int], params: List[torch.Tensor]):
        self.pid_fid = pid_fid
        self.params = params

    def __str__(self) -> str:
        return f"ReceiveParametersAckPDMSG({self.pid_fid}, {self.params})"


# =============================================================================
# Particle Messages
# =============================================================================

class ReceiveFuncMSG(MSG):
    """
    Message for receiving function calls in Particle Communication.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        pid (int): The particle id.
        msg_name (str): The name of the message.
        args (List[Any]): List of arguments for the function call.
    """
    def __init__(self, pid_fid: Tuple[int, int], pid: int, msg_name: str, args: List[Any]):
        self.pid_fid = pid_fid
        self.pid = pid
        self.msg_name = msg_name
        self.args = args

    def __str__(self) -> str:
        return f"ReceiveFuncMSG({self.pid_fid}, {self.pid}, {self.msg_name}, {self.args})"


class ReceiveFuncAckMSG(MSG):
    """
    Acknowledgment message for receiving function calls in Particle Communication.
    """
    def __init__(self):
        pass


class ReceiveGetMSG(MSG):
    """
    Message for receiving particle data in Particle Communication.

    Attributes:
        pid_fid (Tuple[int, int]): Tuple of particle id and function id.
        pid_caller (int): The particle id of the caller.
        pid (int): The particle id.
    """
    def __init__(self, pid_fid: Tuple[int, int], pid_caller: int, pid: int):
        self.pid_fid = pid_fid
        self.pid_caller = pid_caller
        self.pid = pid

    def __str__(self) -> str:
        return f"ReceiveGetMSG({self.pid_fid}, {self.pid_caller}, {self.pid})"


class ReceiveGetAckMSG(MSG):
    """
    Acknowledgment message for receiving particle data in Particle Communication.

    Attributes:
        fid (int): The function id.
        pid (int): The particle id.
        params (List[torch.Tensor]): List of parameters.
        params_grad (List[torch.Tensor]): List of parameter gradients.
    """
    def __init__(self, fid: int, pid: int, params: List[torch.Tensor], params_grad: List[torch.Tensor]):
        self.fid = fid
        self.pid = pid
        self.params = params
        self.params_grad = params_grad

    def __str__(self) -> str:
        return f"ReceiveGetAckMSG({self.fid}, {self.pid}, {self.params}, {self.params_grad})"
