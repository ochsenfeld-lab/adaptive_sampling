import math
import numpy as np
from typing import Tuple
from scipy.stats import kstat
from ..units import *


def gamd_correction_n(
    grid: np.ndarray,
    cv: np.ndarray,
    delta_V: np.ndarray,
    korder: int = 2,
    equil_temp: float = 300.0,
    return_hist: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correction to PMF from GaMD using kth cumulant expansion

    Args:
        grid: grid for reaction coordinate
        cv: trajectory of reaction coordinate
        delta_V: trajectory of GaMD bias potential
        korder: order of cumulant expansion, supported up to 4
        equil_temp: equilibrium temperature

    Returns:
        corr: correction to PMF from GaMD
        hist: histogram of cv on grid
    """
    if korder > 4:
        print(" >>> Warning: cumulant expansion only supportet up to 4th order")
        korder = 4

    beta = 1.0 / (R_in_SI * equil_temp / 1000.0)  # kJ/mol
    dV_kJ = delta_V * atomic_to_kJmol  # kJ/mol

    dx2 = (grid[1] - grid[0]) / 2.0
    hist = np.zeros(len(grid))
    corr = np.zeros(len(grid))
    for i, x in enumerate(grid):
        dat_i = dV_kJ[np.where(np.logical_and(cv >= x - dx2, cv < x + dx2))]
        hist[i] = len(dat_i)
        if hist[i] > 0:
            # kth order cumulant expansion
            for k in range(1, korder + 1):
                corr[i] += (np.power(beta, k) / math.factorial(k)) * kstat(dat_i, k)

    if return_hist:
        return -corr / beta, hist
    else:
        return -corr / beta


def gamd_pmf(
    grid: np.ndarray,
    cv: np.ndarray,
    delta_V: np.ndarray,
    korder: int = 2,
    equil_temp: float = 300.0,
):
    """compute pmf from GaMD simulation
    by correcting probability density with kth order cumulant expansion

    Args:
        grid: grid for reaction coordinate
        cv: trajectory of reaction coordinate
        delta_V: trajectory of GaMD bias potential
        korder: order of cumulant expansion, supported up to 4
        equil_temp: equilibrium temperature

    Returns:
        pmf: potential of mean force
        rho: probability density
    """
    RT = R_in_SI * equil_temp / 1000.0  # kJ/mol
    dx = grid[1] - grid[0]

    # compute pmf from biased prob. density corrected by second order cumulant expansion
    gamd_corr, hist = gamd_correction_n(
        grid,
        cv,
        delta_V,
        korder=korder,
        equil_temp=equil_temp,
        return_hist=True,
    )
    rho = hist / (hist.sum() * dx)
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0)) + gamd_corr

    rho = np.exp(-pmf / RT)
    rho /= rho[~np.isnan(pmf)].sum() * dx
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))

    return pmf, rho
