import numpy as np
from ..units import *
from .utils import *
from ..sampling_tools.utils import diff 

def metapot_from_traj(
    grid: np.ndarray,
    cv: np.ndarray, 
    hill_height: float, 
    hill_std: float,
    hill_drop_freq: int, 
    well_tempered_temp: float=None,
    periodic: bool=False,
):
    """ Get bias potential from metadynamics trajectory
    """
    dx       = grid[1]  - grid[0] 
    minx     = grid[0]  - dx/2.
    maxx     = grid[-1] + dx/2.
    hill_var =  hill_std * hill_std
    hill_height /= atomic_to_kJmol

    metapot = np.zeros_like(grid)
    for i, xi in enumerate(cv):

        if not (minx <= xi <= maxx):
            continue

        grid_idx = int(np.floor(np.abs(xi - minx) / dx))
        
        if i % hill_drop_freq == 0:
        
            if well_tempered_temp != None:
                w = hill_height * np.exp(
                    -metapot[grid_idx]
                    / (kB_in_atomic * well_tempered_temp)
                )
            else:
                w = hill_height
        
            bin_diff = diff(grid, xi, "angle" if periodic else "None")
            metapot += w * np.exp(-(bin_diff * bin_diff) / (2.0 * hill_var))

    return metapot * atomic_to_kJmol
  
def pmf_from_metapot(
    metapot: np.ndarray, 
    well_tempered_temp: float=None, 
    equil_temp: float=300.0
):
    """ Get potential of mean force from the metadyanamics bias potential
    """
    pmf = -metapot
    if well_tempered_temp != None:
        pmf *= (
            equil_temp + well_tempered_temp
        ) / well_tempered_temp
    pmf -= pmf.min()
    return pmf
