import sys
import numpy as np
from dataclasses import dataclass

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


@dataclass
class SamplingData:
    """The necessary sampling data to perform the enhanced sampling."""

    mass: np.ndarray  # Masses in atomic units, shape (natoms,)
    coords: np.ndarray  # Cartesian coordinates in Bohr, shape (3 * natoms,)
    forces: np.ndarray  # Forces in Hartree/Bohr, shape (3 * natoms,)
    epot: float  # Potential energy in Hartree
    temp: float  # Temperature in Kelvin
    natoms: int  # Number of atoms
    step: int  # MD step number
    dt: float  # MD step size in fs


class MDInterface(Protocol):
    def get_sampling_data(self) -> SamplingData:
        """Define this function for your MD class to provide the
        required sampling data for adaptive sampling. If you do not
        wish to have this package as a dependency, wrap the import
        in a `try`/`except` clause, e.g.,

        ```
        class MD:
            # Your MD code
            ...

            def get_sampling_data(self):
                try:
                    from adaptive_sampling.interface.sampling_data import SamplingData

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
                    raise NotImplementedError("`get_sampling_data()` is missing `adaptive_sampling` package") from e
        ```
        """
        ...
