import sys
import numpy as np
from dataclasses import dataclass

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


@dataclass
class SamplingData:
    mass: np.ndarray
    coords: np.ndarray
    forces: np.ndarray
    epot: float
    temp: float
    natoms: int
    step: int
    dt: float


class MDInterface(Protocol):
    def get_sampling_data(self) -> SamplingData:
        pass
