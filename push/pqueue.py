import torch.multiprocessing as mp
from typing import *


class PQueue:
    def __init__(self) -> None:
        pass

    def put(self, msg: any) -> None:
        raise NotImplementedError
    
    def get(self) -> any:
        raise NotImplementedError


class SinglePQueue(PQueue):
    """Single-processing queue.
    """    
    def __init__(self) -> None:
        self._queue = []      # queue
        self._nel = None      # Node Event Loop

    def put(self, msg: any) -> None:
        """Put a message on the queue.

        Args:
            msg (any): Message to send to single-processing queue.
        """
        self._queue += [msg]
        # NOTE: Since put in a multi-processing setting "calls" the function, we have to dispatch here.
        self._nel._dispatch(msg)

    def get(self) -> any:
        """Obtain a message from the queue.

        Returns:
            any: The message.
        """        
        return self._queue.pop()


class MultiPQueue(PQueue):
    """Multi-processing queue.
    """    
    def __init__(self, manager: mp.Manager) -> None:
        self._queue = manager.Queue()

    def put(self, msg: any) -> None:
        """Put a message on the queue. Does not block.

        Args:
            msg (any): Message to send to multi-processing queue.
        """
        self._queue.put(msg)

    def get(self) -> any:
        """Obtain a message from the queue. Blocking operation.

        Returns:
            any: The message.
        """        
        return self._queue.get()
    