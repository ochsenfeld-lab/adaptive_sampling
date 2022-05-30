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
        """Define this function for your MD class to provide the
        required sampling data for adaptive biasing. If you do not
        wish to have this package as a dependency, wrap the import
        in a `try`/`except` clause, e.g.,

        ```
        class MD:
            # Your MD code
            ...

            def get_sampling_data(self):
                try:
                    from adaptive_biasing.interface.sampling_data import SamplingData

                    mass   = ...
                    coords = ...
                    forces = ...
                    epot   = ...
                    temp   = ...
                    natoms = ...
                    step   = ...
                    dt     = ...

                    return SamplingData(mass, coords, forces, epot, temp, natoms, step, dt)
                except ImportError as e:
                    raise NotImplementedError("`get_sampling_data()` is missing `adaptive_biasing` package") from e
        ```
        """
        ...
