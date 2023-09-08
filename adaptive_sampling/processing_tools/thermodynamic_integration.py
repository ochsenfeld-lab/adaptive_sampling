import numpy as np
from ..units import *
from .utils import *

def integrate(
    mean_force: np.ndarray,
    dx: float,
    equil_temp: float = 300.0,
    method: str = "trapezoid",
    normalize: bool = True,
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

    if normalize:
        RT = R_in_SI * equil_temp / 1000.0
        rho = np.exp(-pmf / RT)
        rho /= rho.sum() * dx

        pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))
        return pmf, rho
    else:
        return pmf


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

    # TODO: add periodicity to handle periodic simulations

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

def force_error(
    grid: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    n_bins_tau: int=5,
    equil_temp: float = 300.0,
    full_output: bool = False,
):
    """Error estimation form the standard deviation in an harmonic bias force

    Args:
        grid: grid for reaction coordinate
        xi: trajectory of reaction coordinate
        la: trajectory of fictitious particle
        sigma: thermal width of coupling of la to xi
        n_bins_tau: number of bins for estimation of autocorrelation function
        equil_temp: Temperature of the simulation
        full_output: if True return dict with full information

    returns:
        err_pmf: error estimate for PMF 
    """
    RT = R_in_SI * equil_temp / 1000.0
    k = RT / (sigma * sigma)

    F = -harmonic_force(xi, la, k)

    delta = (xi.max()-xi.min()) / n_bins_tau
    edges = np.arange(xi.min(), xi.max()+delta/2, delta)
    corr = []
    tau  = np.zeros(n_bins_tau)
    for i in range(n_bins_tau):
        frames = np.where(np.logical_and(xi>=edges[i], xi<edges[i+1]))
        corr.append(autocorr(F[frames]))
        tau[i] = ipce(corr[-1])   

    edges_pmf = np.zeros(len(grid)+1)
    edges_pmf[:-1] = grid - (grid[1] - grid[0]) / 2.
    edges_pmf[-1] = edges_pmf[-2] + (grid[1] - grid[0]) / 2.
    
    force     = np.zeros_like(grid)   
    err_force = np.zeros_like(grid)
    tau_pmf   = np.repeat(tau, len(force)//n_bins_tau)
    if (len(force)//n_bins_tau) % 2 != 0:
        tau_pmf = np.append(tau_pmf, tau_pmf[:-1])

    for i in range(len(grid)):
        frames = np.where(np.logical_and(xi>=edges_pmf[i], xi<edges_pmf[i+1]))
        force[i] = F[frames].mean()
        err_force[i] = np.sqrt(tau_pmf[i]/len(F[frames]) * F[frames].var()) 
    
    err_pmf = integrate(err_force, grid[1]-grid[0], normalize=False)

    if full_output:
        return {
            "tau": tau,
            "tau_grid": edges[:-1] + delta/2.,
            "autocorr": corr,
            "force": force,
            "err_force": err_force,
            "err_pmf": err_pmf,
        }
    else:
        return err_pmf


