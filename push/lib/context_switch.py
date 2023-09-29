from typing import *
import threading

import torch
import torch.nn as nn


class ParticleCache:
    """Loads particles on and off the accelerator.
    """    
    def __init__(self, mk_module: Callable, args: List[any], cache_size: int, device: int, lock: threading.Lock) -> None:
        # Module
        self.mk_module = mk_module
        self.args = args

        # Device
        self._device = device           # the device id
        
        # Cache
        self._cache_size = int(cache_size)   # maximum cache size
        self._cache2pid = {}            # cache position -> pid
        self._active2pid = {}           # cache position -> (pid, module)
        self._next_pos = 0

        # All particles
        self._module_disk = {}          # pid -> module (cpu)
        self._particle_disk = {}        # pid -> path (disk)
        self._optim_cache = {}          # pid -> Optimizer
        self._mk_optims = {}            # pid -> closure
        self._pid2cache = {}            # pid -> cache position
        
        self._pinned = set()            # pinned pid
        self._lock = lock               # lock for pinned pid

    def save_to_disk(self, pid, path="./particles"):
        module = self.mk_module(*self.args)
        if pid in self._module_disk:
            params, _ = self._module_disk[pid]
            for p, param in zip(module.parameters(), params):
                p.data = param 
            torch.save(module.state_dict(), self._particle_disk[pid])
        else:
            c_idx = self._pid2cache[pid]
            module = self._active2pid[c_idx][1]
            torch.save(module.state_dict(), self._particle_disk[pid])
        
    def _save_w_grads(self, pid: int, module: nn.Module) -> None:
        params = []
        params_grad = []
        for param in module.parameters():
            params_grad += [param.grad.detach().to("cpu") if param.grad is not None else None]
            params += [param.detach().to("cpu")]
        self._module_disk[pid] = (params, params_grad)

    def _load_w_grads(self, pid: int, module: nn.Module) -> None:
        params, params_grad = self._module_disk[pid]
        for p, param, param_grad in zip(module.parameters(), params, params_grad):
            p.data = param
            p.grad = param_grad
            
    def create(self, pid: int, mk_optim: Callable) -> nn.Module:
        # Create module
        module = self.mk_module(*self.args)
        module = module.to(self._device)
        # Save new module to disk
        self._module_disk[pid] = (module.parameters(), [param.grad for param in module.parameters()])

        c_idx = self._next_pos % self._cache_size
        if c_idx in self._active2pid:
            # There is an active particle sharing the place that we want to write our new particle into
            active_pid, active_module = self._active2pid.pop(c_idx)
            
            # Save active module to disk
            self._save_w_grads(active_pid, active_module)
            
            # Write new module into existing module
            active_module.load_state_dict(module.state_dict())
            
            # Update
            self._active2pid[c_idx] = (pid, active_module)
            module = active_module
        else:
            self._cache2pid[c_idx] = set()
            self._active2pid[c_idx] = (pid, module)
        # module should be the one in self._active2pid

        self._cache2pid[c_idx].add(pid)
        self._pid2cache[pid] = c_idx

        # Put in cache
        self._particle_disk[pid] = f"particles/device{self._device}_particle{pid}.pth"
        self._mk_optims[pid] = mk_optim
        self._optim_cache[pid] = mk_optim(module.parameters())

        # Increment
        self._next_pos += 1
        
        return module

    def try_pin(self, pid: int) -> bool:
        c_idx = self._pid2cache[pid]
        with self._lock:
            if self._active2pid[c_idx][0] == pid:
                self._pinned.add(pid)
                return True
            else:
                return False

    def release(self, pid, thread):
        with self._lock:
            thread.join()
            if pid in self._pinned:
                self._pinned.remove(pid)

    def unpin(self, pid: int) -> None:
        with self._lock:
            if pid in self._pinned:
                self._pinned.remove(pid)

    def try_read(self, pid: int, pin=False, msg=None) -> nn.Module:
        # if msg is not None:
        #     print(msg)
        c_idx = self._pid2cache[pid]
        with self._lock:
            if self._active2pid[c_idx][0] == pid:
                # Return if's active
                if pin:
                    self._pinned.add(pid)
                return self._active2pid[c_idx][1]
            elif self._active2pid[c_idx][0] in self._pinned:
                # Return None if its pinned
                return None
            
            # Pin
            if pin:
                self._pinned.add(pid)

            # Prepare to swap
            active_pid, active_module = self._active2pid.pop(c_idx)

            # Save with gradients
            self._save_w_grads(active_pid, active_module)

            # Remove old optimizer
            old_optim = self._optim_cache.pop(active_pid)
            del old_optim 

            # Load particle into active module
            self._load_w_grads(pid, active_module)
            self._active2pid[c_idx] = (pid, active_module.to(self._device))            
            
            # Return new module
            new_module = self._active2pid[c_idx][1]
            
            # Restore optim
            self._optim_cache[pid] = self._mk_optims[pid](new_module.parameters())
            
            # Result
            return new_module

    def contains(self, pid):
        return pid in self._pid2cache

    def particles(self) -> List[int]:
        return self._pid2cache.keys()
    
    def __str__(self) -> str:
        return f"active2pid: {str({k: v[0] for k, v in self._active2pid.items()})}\n cache2pid: {str(self._cache2pid)}"
    

class ParticleCacheLRU:
    """Loads particles on and off the accelerator.
    """    
    def __init__(self, mk_module: Callable, args: List[any], cache_size: int, device: int) -> None:
        # Module
        self.mk_module = mk_module
        self.args = args

        # Device
        self._device = device           # the device id
        
        # Cache
        self._cache_size = cache_size   # maximum cache size
        self._module_cache = {}         # pid -> module (device)
        self._module_disk = {}          # pid -> module (cpu)
        self._particle_disk = {}        # pid -> path (disk)
        self._optim_cache = {}          # pid -> Optimizer
        self._mk_optims = {}            # pid -> closure
        self._lru = []

    def _save(self, pid: int, module: nn.Module, disk=False) -> None:
        if disk:
            torch.save(module.state_dict(), self._particle_disk[pid])
        else:
            tmp = self.mk_module(*self.args)
            tmp.load_state_dict(module.state_dict())
            self._module_disk[pid] = tmp

    def _load(self, pid: int, module: nn.Module, disk=False) -> None:
        if disk:
            checkpoint = torch.load(self._particle_disk[pid])
            module.load_state_dict(checkpoint)
        else:
            module.load_state_dict(self._module_disk[pid].state_dict())
            
    def read(self, pid: int) -> nn.Module:
        if pid in self._module_cache:
            return self._module_cache[pid]
        else:
            if len(self._module_cache) >= self._cache_size:
                # Remove particle
                lru_idx = self._lru.pop(0)
                module = self._module_cache.pop(lru_idx)
                self._save(lru_idx, module)
        
                # Remove old optimizer
                old_optim = self._optim_cache.pop(pid)
                del old_optim 

                # Load particle
                self._load(pid, module)
                self._module_cache[pid] = module.to(self._device)
                self._lru += [pid]
                new_module = self._module_cache[pid]
                
                # Restore optim
                self._optim_cache[pid] = self._mk_optims[pid](new_module.parameters())
                
                # Result
                return new_module
            else:
                raise ValueError("Shouldn't happen ...")

    def write(self, pid: int, module: nn.Module) -> None:
        if pid in self._module_cache:
            self._module_cache[pid] = module
        else:
            if len(self._module_cache) >= self._cache_size:
                lru_idx = self._lru.pop(0)
                module_p = self._module_cache.pop(lru_idx)
                self._save(lru_idx, module_p)

            self._module_cache[pid] = module.to(self._device)
            self._lru += [pid]

    def create(self, pid: int, mk_optim: Callable) -> nn.Module:
        self._particle_disk[pid] = f"particles/device{self._device}_particle{pid}.pth"
        module = self.mk_module(*self.args)
        module = module.to(self._device)
        self.write(pid, module)
        
        self._mk_optims[pid] = mk_optim
        self._optim_cache[pid] = mk_optim(module.parameters())

        return module

    def contains(self, pid):
        return pid in self._module_cache

    def particles(self) -> List[int]:
        return self._module_cache.keys()
