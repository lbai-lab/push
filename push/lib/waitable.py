from typing import *


class Waitable:
    def __init__(self):
        pass
    
    def _wait(self, fid: int) -> any:
        raise NotImplementedError
    