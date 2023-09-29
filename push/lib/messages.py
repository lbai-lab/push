import torch
from typing import *


class MSG:
    pass


# =============================================================================
# Device Event Loop Messages
# =============================================================================

class DevEvtLoopInitMSG(MSG):
    def __init__(self):
        pass


class DevEvtLoopCleanupMSG(MSG):
    def __init__(self):
        pass


class DELBroadcastParticlesMSG(MSG):
    def __init__(self, in_queues, out_queues, particle_to_device):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.particle_to_device = particle_to_device


class DELBroadcastParticlesAckMSG(MSG):
    def __init__(self):
        pass


class NELSaveModel(MSG):
    def __init__(self, pid_fid):
        self.pid_fid = pid_fid


class NELSaveModelAckPDMSG(MSG):
    def __init__(self, pid_fid):
        self.pid_fid = pid_fid


# =============================================================================
# Particle Neural Network Messages
# =============================================================================

# -----------------------------------------------------
# One-Time
# -----------------------------------------------------

class ReceiveParticleInitPNNMSG(MSG):
    def __init__(self, device: int, pid: int, mk_optim: Callable, receive, state):
        self.device = device
        self.pid = pid
        self.mk_optim = mk_optim
        self.receive = receive
        self.state = state


class ReceiveParticleInitAckPNNMSG(MSG):
    def __init__(self):
        pass


class ReceiveRegisterPNNMSG(MSG):
    def __init__(self, pid: int, msg: str, fn: Callable, state: dict[str, any]):
        self.pid = pid
        self.msg = msg
        self.fn = fn
        self.state = state


class ReceiveRegisterAckPNNMSG(MSG):
    def __init__(self):
        pass


# -----------------------------------------------------
# Multi-Time
# -----------------------------------------------------

class ReceiveFuncPNNMSG(MSG):
    def __init__(self, pid_fid, pid_to: int, msg: str, args: list[any]):
        self.pid_fid = pid_fid
        self.pid_to = pid_to
        self.msg = msg
        self.args = args


class ReceiveFuncAckPNNMSG(MSG):
    def __init__(self, pid_fid):
        self.pid_fid = pid_fid


class ReceiveParametersPNNMSG(MSG):
    def __init__(self, pid_fid, pid: int):
        self.pid_fid = pid_fid
        self.pid = pid


class ReceiveParametersAckPNNMSG(MSG):
    def __init__(self, pid_fid, params: list[torch.Tensor]):
        self.pid_fid = pid_fid
        self.params = params


# =============================================================================
# Particle Messages
# =============================================================================

class ReceiveFuncMSG(MSG):
    def __init__(self, pid_fid, pid: int, msg_name: str, args: list[any]):
        self.pid_fid = pid_fid
        self.pid = pid
        self.msg_name = msg_name
        self.args = args


class ReceiveFuncAckMSG(MSG):
    def __init__(self):
        pass


class ReceiveGetMSG(MSG):
    def __init__(self, pid_fid, pid_caller: int, pid: int):
        self.pid_fid = pid_fid
        self.pid_caller = pid_caller
        self.pid = pid


class ReceiveGetAckMSG(MSG):
    def __init__(self, fid: int, pid: int, params: list[torch.Tensor], params_grad: list[torch.Tensor]):
        self.fid = fid
        self.pid = pid
        self.params = params
        self.params_grad = params_grad
