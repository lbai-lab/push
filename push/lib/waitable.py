from typing import *


class Waitable:
    """
    An abstract class representing an object with a waitable operation.

    Attributes:
        None

    Methods:
        _wait(fid) -> Any: 
            Wait for the completion of a specific operation identified by the future ID (fid).
            This method should be implemented by subclasses.

    Usage:
        This class can be subclassed to create objects that support waitable operations.
    """

    def __init__(self):
        """Initialize the Waitable object."""
        pass

    def _wait(self, fid: int) -> Any:
        """Wait for the completion of a specific operation identified by the future ID (fid).

        Args:
            fid (int): The future ID representing the operation.

        Returns:
            Any: The result of the completed operation.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError