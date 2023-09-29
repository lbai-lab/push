import threading
from typing import *

from push.lib.waitable import Waitable


class PFuture:
    """Particle Future. Eventually contains a result.
    """    
    def __init__(self, waitable: Waitable, send_pid: int, recv_pid: int, fid: int, t: Union[None, threading.Thread]=None):
        self._waitable = waitable   # waitable control flow
        self._send_pid = send_pid   # pid of particle that sends message
        self._recv_pid = recv_pid   # pid of particle that receives message and contains result
        self._fid = fid             # unique future identifier
        self._t_or_ts = t           # potential thread to block on

    def wait(self) -> any:
        """Wait until we get a result.

        Returns:
            any: The value that we waited to get.
        """        
        if self._t_or_ts is None:
            return self._waitable._wait(self._fid)
        elif isinstance(self._t_or_ts, threading.Thread):
            self._waitable._wait_particle_thread(self._recv_pid)
            return self._waitable._wait(self._fid)
        else:
            raise ValueError("Shouldn't happen ...")

    def __eq__(self, other) -> bool:
        return isinstance(other, PFuture) and self._fid == other._fid

    def __hash__(self) -> int:
        return self._fid
