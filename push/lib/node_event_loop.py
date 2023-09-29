from collections import OrderedDict
import threading
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import *
import time

from push.lib.context_switch import ParticleCacheLRU, ParticleCache
from push.lib.messages import *
from push.lib.utils import detach_to_device, detach_to_cpu
from push.lib.waitable import Waitable
from push.particle import Particle, ParticleView
from push.pfuture import PFuture


class NodeEventLoop(Waitable):
    """
    1. Each node event loop is mapped to all devices on that node.
    2. The node event loop is responsible for executing all operations on particles.
    """    
    def __init__(self,
                 mk_module: Callable,
                 args: list[any],
                 in_queue: mp.Queue,
                 out_queue: mp.Queue,
                 rank: int,
                 devices: int,
                 cache_size: int,
                 view_size: int) -> None:
        # Node information
        self.rank = rank                         # Rank of NEL
        self.devices = devices                   # Devices on this NEL
        
        # Messaging back out to PusH
        self.in_queue = in_queue                 # receiving message queue
        self.out_queue = out_queue               # direct queue to PusH

        # Point-to-point messaging between particles
        self._in_queues = {}                     # Type: pid -> mp.Queue
        self._out_queues = {}                    # Type: pid -> mp.Queue

        # Particle information
        self._particle_to_rank = {}              # Type: pid -> rank
        self._particle_to_device = {}            # Type: pid -> device
        self._particle_to_state = {}             # Type: pid -> state
        self._hooks = {}                         # Type: pid -> msg -> closure, for triggering events

        # Device management
        self._device_to_pthread = OrderedDict()  # Type: device -> (pid, thread)
        self.particle_caches = {}                # Type: device -> cache, particle caches on this NEL
        self.view_caches = {}                    # Type: device -> cache, cross-device view caches on this NEL
        for device in devices:
            self.particle_caches[device] = ParticleCache(mk_module, args, cache_size, device, threading.Lock())
            self.view_caches[device] = ParticleCache(mk_module, args, view_size, device, threading.Lock())
        
        # Futures state for particles on this event loop
        self._future_id = 0                      # Type: fid, future id
        self._particle_to_futures = {}           # Type: pid -> set[fid]
        self._future_to_particle = {}            # Type: fid -> pid
        
        self._results = {}                       # Type: fid -> result
        self._results_lock = threading.Lock()

    # -----------------------------------------------------
    # Context switching
    # -----------------------------------------------------

    def _wait_particle_thread(self, pid: int) -> None:
        pid_device = self._particle_to_device[pid]
        if pid_device in self._device_to_pthread:
            # If device is in use, finish execution of active particle that is using that device
            active_pid, active_thread = self._device_to_pthread.pop(pid_device)
            active_pid_device = self._particle_to_device[active_pid]
            self.particle_caches[active_pid_device].release(active_pid, active_thread)

    def _context_switch_module(self, pid: int, pin=False, msg=None) -> nn.Module:
        pid_device = self._particle_to_device[pid]
        
        # Try read
        self._wait_particle_thread(pid)
        module = self.particle_caches[pid_device].try_read(pid, pin=pin, msg=msg)
        
        # Try reload module
        while module is None:
            # If module read fails, finish execution of active particle that is using that device
            active_pid, active_thread = self._device_to_pthread.popitem(last=False)
            active_pid_device = self._particle_to_device[active_pid]
            self.particle_caches[active_pid_device].release(active_pid, active_thread)
            
            # Try reload module
            module = self.particle_caches[pid_device].try_read(pid, pin=pin)
        return module

    # -----------------------------------------------------
    # Node event loop entry
    # -----------------------------------------------------

    def _start_event_loop(self) -> None:
        go = True
        while go:
            # Obtain message and dispatch
            msg = self.in_queue.get()
            go = self._dispatch(msg)

    def _dispatch(self, msg: MSG) -> bool:

        # -----------------------------------------------------
        # Particle functionality
        # -----------------------------------------------------
        
        if isinstance(msg, ReceiveFuncMSG):
            # Handle message
            sender, pid, msg_name = msg.pid_fid, msg.pid, msg.msg_name
            if msg_name in self._hooks[pid]:
                fn = self._hooks[pid][msg_name]
                state = self._particle_to_state[pid]
                pid_device = self._particle_to_device[pid]
                self._context_switch_module(pid, msg='ReceiveFuncMsg')
                particle = Particle(self, pid_device, pid, module, state)
                args = [detach_to_device(pid_device, arg) for arg in msg.args]
                fn(particle, *args)

        elif isinstance(msg, ReceiveGetMSG):
            # Handle message
            pid_device = self._particle_to_device[msg.pid]
            module = self._context_switch_module(pid)
            params = []
            params_grad = []
            for p in module.parameters():
                params += [p.detach().clone().cpu()]
                params_grad += [p.grad.detach().clone().cpu() if p.grad is not None else None]
            
            # Acknowledge
            caller_pid_device = self._particle_to_device[msg.pid_caller]
            self._in_queues[caller_pid_device].put(ReceiveGetAckMSG(msg.pid_fid, msg.pid, params, params_grad))

        # -----------------------------------------------------
        # PNN functionality
        # -----------------------------------------------------

        elif isinstance(msg, ReceiveParametersPDMSG):
            # Handle message
            pid_device = self._particle_to_device[msg.pid]
            module = self._context_switch_module(msg.pid)
            params = [x.detach().to("cpu") for x in module.parameters()]

            # Acknowledge
            self.out_queue.put(ReceiveParametersAckPDMSG(msg.pid_fid, params))

        elif isinstance(msg, ReceiveRegisterPDMSG):
            # Handle message
            self._hooks[msg.pid][msg.msg] = msg.fn

            # Acknowledge
            self.out_queue.put(ReceiveRegisterAckPDMSG())

        elif isinstance(msg, ReceiveParticleInitPDMSG):
            # Handle message
            self._particle_to_rank[msg.pid] = self.rank       # set rank
            self._particle_to_device[msg.pid] = msg.device    # set device
            module = self.particle_caches[self._particle_to_device[msg.pid]].create(msg.pid, msg.mk_optim)
            self._particle_to_state[msg.pid] = msg.state
            
            # Register receives
            self._hooks[msg.pid] = {}
            for recv_msg, fn in msg.receive.items():
                self._hooks[msg.pid][recv_msg] = fn
            self._particle_to_futures[msg.pid] = set()

            # Acknowledge
            self.out_queue.put(ReceiveParticleInitAckPDMSG())

        elif isinstance(msg, NELSaveModel):
            # Handle message
            for pid, pid_device in self._particle_to_device.items():
                self.particle_caches[pid_device].save_to_disk(pid)

            # Acknowledge
            self.out_queue.put(NELSaveModelAckPDMSG(msg.pid_fid))

        # -----------------------------------------------------
        # Node Event Loop Functionality
        # -----------------------------------------------------

        elif isinstance(msg, NELBroadcastParticlesMSG):
            # Handle message
            self._in_queues = msg.in_queues
            self._out_queues = msg.out_queues
            self._particle_to_device = msg.particle_to_device

            # Acknowledge
            self.out_queue.put(NELBroadcastParticlesAckMSG())

        elif isinstance(msg, NodeEvtLoopCleanupMSG):
            return False

        # -----------------------------------------------------
        # PNN - particle functionality
        # -----------------------------------------------------
        
        elif isinstance(msg, ReceiveFuncPDMSG):
            # Handle message
            sender, pid, msg_name = msg.pid_fid, msg.pid_to, msg.msg
            if msg_name in self._hooks[pid]:
                # fn, state = self._hooks[pid][msg_name]
                fn = self._hooks[pid][msg_name]
                state = self._particle_to_state[pid]
                pid_device = self._particle_to_device[pid]
                module = self._context_switch_module(pid)
                particle = Particle(self, pid_device, pid, module, state)
                # fn(particle, *msg.args)
                try:
                    y = fn(particle, *msg.args)
                except Exception as e:
                    self.out_queue.put(e)
                    self._cleanup()
                    raise e
                
                # Acknowledge
                self.out_queue.put(ReceiveFuncAckPDMSG(sender))

        return True        

    def _dispatch_receive_get_ack(self, msg: ReceiveGetAckMSG) -> None:
        # Create particle on this device event loop if it doesn't exist
        pid_device = self._particle_to_device[msg.pid]
        if not self.view_caches[pid_device].contains(msg.pid):
            module = self.view_caches[pid_device].create(msg.pid, lambda x: None)
        else:
            module = self.view_caches[pid_device].read(msg.pid)

        # Copy parameters over
        for param, p, p_grad in zip(module.parameters(), msg.params, msg.params_grad):
            with torch.no_grad():
                param.copy_(p)
            if p_grad is not None:
                param.grad = p_grad.to(pid_device)

        # Return a lazy view
        self._results[msg.fid] = ParticleView(self.view_caches[self._particle_to_device[msg.pid]], msg.pid)

    def _wait(self, fid: int) -> any:
        def check_results(fid: int):
            # Check to see if we already have the result
            if fid in self._results:
                result = self._results.pop(fid)
                pid = self._future_to_particle.pop(fid)
                self._particle_to_futures[pid].remove(fid)
                return True, result
            else:
                return False, None

        result = check_results(fid)
        msgs = []
        while not result[0]:
            msg = self.in_queue.get()
            if isinstance(msg, ReceiveGetAckMSG):
                self._dispatch_receive_get_ack(msg)
                result = check_results(fid)
            elif isinstance(msg, ReceiveFuncMSG) or isinstance(msg, ReceiveFuncPDMSG):
                # Note: Handle this particles receive
                #       Additional parallelism here if we are not on the same PID
                msgs += [msg]
            else:
                # Dispatch PNN messages
                go = self._dispatch(msg)
        
        # Dispatch buffered messages
        for msg in msgs:
            go = self._dispatch(msg)
        
        return result[1]

    def _cleanup(self) -> None:
        self._dispatch(NodeEvtLoopCleanupMSG())

    # -----------------------------------------------------
    # Future functionality
    # -----------------------------------------------------

    def _create_future_id(self) -> int:
        fid = self._future_id
        self._future_id += 1
        return fid

    def _register_future(self, pid: int, fid: int) -> None:
        self._particle_to_futures[pid].add(fid)
        self._future_to_particle[fid] = pid

    # -----------------------------------------------------
    # Particle functionality
    # -----------------------------------------------------

    def particles(self) -> List[int]:
        return list(self._particle_to_device.keys())

    def register_receive(self, pid: int, msg:str, fn: Callable, state: dict) -> None:
        self._hooks[pid][msg] = (fn, state)

    def send(self, send_particle: Particle, pid_curr: int, pid: int, msg_name: str, *args: any) -> PFuture:
        # Create future
        fid = self._create_future_id()
        self._register_future(pid_curr, fid)

        # Check communication
        rank_id_curr = self._particle_to_rank[pid_curr]
        rank_id = self._particle_to_rank[pid]
        if rank_id == rank_id_curr: # We are on the same rank
            # NOTE: INVARIANT: the current function always has it's module in scope
            module = self._context_switch_module(pid, msg=f'send {msg_name} from {pid_curr} to {pid}')
            
            # NOTE: Execute function on main node event loop.
            #       Compute heavy items are run on separate threads.
            # fn, state = self._hooks[pid][msg_name]
            fn = self._hooks[pid][msg_name]
            state = self._particle_to_state[pid]
            pid_device = self._particle_to_device[pid]
            particle = Particle(self, pid_device, pid, module, state)
            args2 = [detach_to_device(pid_device, arg) for arg in args]
            try:
                y = fn(particle, *args2)
            except Exception as e:
                self.out_queue.put(e)
                self._cleanup()
                raise e
            self._results[fid] = y

            # NOTE: INVARIANT: the current function always has it's module in scope
            send_particle.module = self._context_switch_module(pid_curr, msg='send switch back')

            # Return future
            return PFuture(self, pid_curr, pid, fid)
        else:
            # Need to communicate
            args2 = [detach_to_cpu(arg) for arg in args]
            self._in_queues[rank_id].put(ReceiveFuncMSG((pid_curr, fid), pid, msg_name, args2))
            
            # Return future
            return PFuture(self, pid_curr, pid, fid)

    def get(self, pid_curr: int, pid: int) -> PFuture:
        # Create future
        fid = self._create_future_id()
        self._register_future(pid_curr, fid)

        # Check communication
        rank_id_curr = self._particle_to_rank[pid_curr]
        rank_id = self._particle_to_rank[pid]        
        if rank_id == rank_id_curr: # We are on the same rank
            # NOTE: INVARIANT: the current function always has it's module in scope
            module = self._context_switch_module(pid, msg='get')

            # Create space for pid on pid_curr's device
            device_curr = self._particle_to_device[pid_curr]
            if not self.view_caches[device_curr].contains(pid):
                module_on_curr = self.view_caches[device_curr].create(pid, lambda x: None)
            else:
                module_on_curr = self.view_caches[device_curr].try_read(pid)

            if True:
                # Copy parameters over
                for param, p in zip(module_on_curr.parameters(), module.parameters()):
                    with torch.no_grad():
                        param.copy_(p).to(device_curr)
                    if p.grad is not None:
                        param.grad = p.grad.to(device_curr)

                self._results[fid] = ParticleView(self.view_caches[device_curr], pid)
                return PFuture(self, pid_curr, pid, fid)
            else:
                def worker(device_curr, module_on_curr, module):
                    for param, p in zip(module_on_curr.parameters(), module.parameters()):
                        with torch.no_grad():
                            param.copy_(p).to(device_curr)
                        if p.grad is not None:
                            param.grad = p.grad.to(device_curr)

                        self._results[fid] = ParticleView(self.view_caches[device_curr], pid)

                # Launch thread
                t = threading.Thread(target=worker, args=(device_curr, module_on_curr, module,))
                self._device_to_pthread[device_curr] = (pid_curr, t)
                t.start()
                # Return future
                return PFuture(self, pid_curr, pid, fid, t=t)
        else:
            # Need to communicate
            self._in_queues[rank_id].put(ReceiveGetMSG(fid, pid_curr, pid))

            # Return future
            return PFuture(self, pid_curr, pid, fid)

    def wait(self, pfutures: List[PFuture]) -> List[any]:
        acc = []
        for pfuture in pfutures:
            acc += [pfuture.wait()]
        return acc

    # -----------------------------------------------------
    # NN Functionality
    # -----------------------------------------------------

    def zero_grad(self, pid: int) -> PFuture:
        # Register future
        fid = self._create_future_id()
        self._register_future(pid, fid)

        def worker(module):
            module.zero_grad()
            self._results[fid] = None

        # Context switch
        module = self._context_switch_module(pid, msg='zero_grad')
        # # Functionality for zero grad
        # module.zero_grad()
        # self._results[fid] = None

        # Launch thread
        t = threading.Thread(target=worker, args=(module,))
        pid_device = self._particle_to_device[pid]
        self._device_to_pthread[pid_device] = (pid, t)
        t.start()

        # Return future
        return PFuture(self, pid, pid, fid, t=t)

    def forward(self, pid: int, x: torch.Tensor, *args: any) -> PFuture:
        # Register future
        fid = self._create_future_id()
        self._register_future(pid, fid)

        # Functionality for forward
        def worker(module):
            try:
                y = module.forward(x, *args)
                self._results[fid] = y
            except Exception as e:
                self.out_queue.put(e)
                self._cleanup()
                raise e

        # Context switch
        module = self._context_switch_module(pid, msg='forward')
        # y = module.forward(x, *args)
        # self._results[fid] = y

        # Launch thread
        t = threading.Thread(target=worker, args=(module,))
        pid_device = self._particle_to_device[pid]
        self._device_to_pthread[pid_device] = (pid, t)
        t.start()
        
        # Return future
        return PFuture(self, pid, pid, fid, t=t)

    def step(self, pid: int, loss_fn: Callable, data: torch.Tensor, label: torch.Tensor, *args: any) -> PFuture:
        # Register future
        fid = self._create_future_id()
        self._register_future(pid, fid)
        
        # Functionality for step
        def worker(module):
            try:
                # Backwards pass
                module.zero_grad()
                y = module.forward(data, *args)
                loss = loss_fn(y, label)
                loss.backward()

                # Optionally step
                pid_device = self._particle_to_device[pid]
                optim = self.particle_caches[pid_device]._optim_cache[pid]
                if optim:
                    optim.step()

                # NOTE: Assumes that dictionary stores are atomic
                self._results[fid] = loss
            except Exception as e:
                self.out_queue.put(e)
                self._cleanup()
                raise e

        # Context switch
        module = self._context_switch_module(pid, pin=True, msg=f'step {pid}')
        
        # Launch thread
        t = threading.Thread(target=worker, args=(module,))
        pid_device = self._particle_to_device[pid]
        self._device_to_pthread[pid_device] = (pid, t)
        t.start()

        # Return future
        return PFuture(self, pid, pid, fid, t=t)
    