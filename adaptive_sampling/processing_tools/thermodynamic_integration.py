import numpy as np
from ..units import *


def integrate(
    mean_force: np.ndarray,
    dx: float,
    equil_temp: float = 300.0,
    method: str = "trapezoid",
) -> tuple:
    """numeric integration of thermodynamic force by simpson, trapezoid or rectangle rule

    Args:
        data: data to integrate
        dx: bin width
        equil_temp: equilibrium temperature
        method: use 'simpson', 'trapezoid' or 'rectangle' rule

    Returns:
        pmf (np.ndarray): potential of mean force
        rho (np.ndarray): probability density
    """
    RT = R_in_SI * equil_temp / 1000.0
    data = np.copy(mean_force)

    if method == "simpson":
        data[1:-1] *= 2.0
        data[2:-1:2] *= 2.0
        pmf = np.full(len(data), dx / 3.0)
    elif method == "trapezoid":
        data[1:-1] *= 2.0
        pmf = np.full(len(data), dx / 2.0)
    else:  # method == 'rectangle':
        pmf = np.full(len(data), dx)

    for i in range(len(pmf)):
        pmf[i] *= data[0 : i + 1].sum()

    rho = np.exp(-pmf / RT)
    rho /= rho.sum() * dx

    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))
    return pmf, rho


def czar(
    grid: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    equil_temp: float = 300.0,
) -> np.ndarray:
    """Corrected z-averaged Restrained (CZAR)

       see: Lesage et. al., J. Phys. Chem. B (2017); https://doi.org/10.1021/acs.jpcb.6b10055

    Args:
        grid: grid for reaction coordinate
        xi: trajectory of reaction coordinate
        la: trajectory of fictitious particle
        sigma: thermal width of coupling of la to xi
        equil_temp: Temperature of the simulation

    Returns:
        mean_force: thermodynamic force (gradient of PMF)
    """
    RT = R_in_SI * equil_temp / 1000.0

    dx2 = (grid[1] - grid[0]) / 2.0
    grid_local = grid + dx2

    # get force constant from sigma
    k = RT / (sigma * sigma)

    hist = np.zeros(len(grid), dtype=float)
    f_corr = np.zeros(len(grid), dtype=float)
    for i, x in enumerate(grid_local):
        la_x = la[np.where(np.logical_and(xi >= x - dx2, xi < x + dx2))]
        hist[i] = len(la_x)
        if hist[i] > 0:
            la_avg = np.average(la_x)
            f_corr[i] = k * (la_avg - x)

    log_hist = np.log(hist, out=np.zeros_like(hist), where=(hist != 0))

    return -RT * np.gradient(log_hist, grid_local) + f_corr 
