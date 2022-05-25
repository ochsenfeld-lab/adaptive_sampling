import os, sys, math, logging
import numpy as np
from typing import Union, List, Tuple
from ..sampling_tools.units import *
from ..units import *

######################################################################
# helper functions
######################################################################
def rolling_avg(traj: np.ndarray, n: int = 50) -> np.ndarray:
    """calculates the rolling average of a time trajectory"""
    return np.convolve(traj, np.ones(n), "valid") / n


def harmonic(
    x: Union[float, np.ndarray], x0: Union[float, np.ndarray], kx: float
) -> Union[float, np.ndarray]:
    """harmonic potential as implemented in the adaptive biasing module"""
    return 0.5 * kx * (np.power((x - x0), 2))


def harmonic_force(
    x: Union[float, np.ndarray], x0: Union[float, np.ndarray], kx: float
) -> Union[float, np.ndarray]:
    """harmonic force"""
    return kx * (x - x0)


def boltzmann(u_pot: Union[float, np.ndarray], beta: float) -> Union[float, np.ndarray]:
    """get Boltzmann factors"""
    return np.exp(-beta * u_pot, dtype=float)


def join_frames(traj_list: List[np.array]) -> Tuple[np.ndarray, float, np.ndarray]:
    """get one array with all frames from list of trajectories"""
    num_frames = 0
    frames_per_traj = []
    all_frames = traj_list[0]
    for ii, traj in enumerate(traj_list):
        frames_per_traj.append(len(traj))
        num_frames += frames_per_traj[ii]
        if ii > 0:
            all_frames = np.concatenate((all_frames, traj))

    frames_per_traj = np.array(frames_per_traj)
    return all_frames, num_frames, frames_per_traj


def get_us_windows(
    centers: np.ndarray, xi: np.ndarray, la: np.ndarray, sigma: float, T: float = 300.0
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """generate US windows from eABF trajectory

    args:
        centers (np.ndarray): centers of Umbrella windows
        xi (np.ndarray): Trajectory of the reaction coordinate
        la (np.ndarray): Trajectory of the extended variable
        sigma (float): Thermal width of coupling of xi and la
        T (float): equillibrium temperature

    returns:
        traj_list (list): list of window trajectories,
        index_list (np.ndarray): list of frame indices in original trajectory,
        meta_f (np.ndarray): window information for MBAR
    """
    RT = R_in_SI * T / 1000.0
    k = RT / (sigma * sigma)

    dx = centers[1] - centers[0]
    dx2 = dx / 2.0

    traj_list = []
    index_list = np.array([])
    for i, center in enumerate(centers):
        indices = np.where(np.logical_and(la >= center - dx2, la < center + dx2))
        index_list = np.append(index_list, indices[0])
        traj_list += [xi[indices]]

    meta_f = np.zeros(shape=(len(centers), 3))
    meta_f[:, 1] = centers
    meta_f[:, 2] = k

    return traj_list, index_list.astype(np.int32), meta_f


def write_us_data(
    centers: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    E_pot=None,
    T: float = 300.0,
    out: str = "umbrella_data/",
):
    """Get US trajectories and metafile from eABF data
    compatible with most popular WHAM/MBAR/EMUS implementations
    """
    if not os.path.isdir(out):
        os.mkdir(out)
    os.chdir(out)

    traj_list, _, meta_f = get_us_windows(centers, xi, la, sigma, T=T)

    for i in range(len(centers)):
        out = open("cv_meta.dat", "a")
        out.write("umbrella_%d.dat\t%14.6f\t%14.6f\n" % (i, meta_f[i, 1], meta_f[i, 2]))
        out.close()

        timeseries = open("umbrella_%d.dat" % i, "w")
        for t, cv in enumerate(traj_list[i]):
            timeseries.write("%d\t%14.6f\n" % (t, cv))
        timeseries.close()

    if E_pot is not None:
        E_list, _, _ = get_us_windows(centers, E_pot, la, sigma, T=T)
        for i in range(len(centers)):
            e_series = open("epot_%d.dat" % i, "w")
            for t, epot in enumerate(E_list[i]):
                e_series.write("%d\t%14.6f\n" % (t, epot))
            e_series.close()


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm

    args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample

    returns:
        mean (float)
        M2 (float)
        var (float)
    """
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var


def reaction_freeE(
        pmf: np.ndarray, T: float = 300.0, min_bin: int = 20, max_bin: int = -20, TS: int = None
) -> tuple:
    """calculate free energy difference

    args:
        pmf: potential of mean force (free energy surface)
        T: temperature
        min_bin/max_bin: minimum/maximum bin for search of transition state
        TS: alternatively, bin number of TS
    returns:
        dA (float): free energy difference
        dA_grid (np.ndarray): free energy difference on grid
    """
    RT = (R_in_SI * T) / 1000.0
    pmf = pmf[~np.isnan(pmf)]
    
    if TS == None:
        TS = np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0]

    P = np.exp(-pmf / RT)
    P /= P.sum()

    P_a = P[:TS].sum()
    P_b = P[(TS + 1) :].sum()

    dA = RT * np.log(P_a / P_b)
    dA_grid = np.zeros(len(pmf))
    dA_grid[TS] += dA / 2
    dA_grid[(TS + 1) :] += dA
    dA_approx = pmf[TS:].min() - pmf[:TS].min()

    return dA, dA_grid, dA_approx

def activation_freeE(
        pmf: np.ndarray, m_xi_inv: np.ndarray, T: float = 300.0, min_bin: int = 20, max_bin: int = -20, TS: int = None
) -> tuple:
    """calculate activation free energy 

    args:
        pmf: potential of mean force (free energy surface)
        m_xi_inv: z-conditioned average of inverse mass associates with CV, expected units are xi^2/(au_mass * angstrom^2)
        T: temperature
        min_bin/max_bin: minimum/maximum bin for search of transition state
        TS: alternatively, bin number of TS
    returns:
        dA (float): free energy difference
        dA_grid (np.ndarray): free energy difference on grid
    """
    RT = (R_in_SI * T) / 1000.0  # kJ/mol
    
    pmf = fep[~np.isnan(fep)]
    if TS == None:
        TS = np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0]
    
    rho = np.exp(-pmf/RT)
    P = rho / rho.sum() # normalize so that P_a + P_b = 1.0
    
    lambda_xi = np.sqrt((h_in_SI * h_in_SI * m_xi_inv[TS]) / (2. * np.pi * atomic_to_kg  * kB_in_SI * T))
    lambda_xi *= 1e10

    P_a = P[:TS].sum()
    P_b = P[(TS+1):].sum()

    dA_a2b = - RT*np.log((rho[TS]*lambda_xi) / P_a)
    dA_b2a = - RT*np.log((rho[TS]*lambda_xi) / P_b)
    dA_a2b_approx = pmf[TS] - pmf[:TS].min()
    dA_b2a_approx = pmf[TS] - pmf[TS:].min()
    
    return dA_a2b, dA_b2a, dA_a2b_approx, dA_b2a_approx 



######################################################################
# Corrected Z-Averaged Restrained (Lesage et al. 2016)
######################################################################
def CZAR(
    grid: np.ndarray, xi: np.ndarray, la: np.ndarray, sigma: float, T: float = 300.0
) -> np.ndarray:
    """Corrected z-averaged restrained

    args:
        grid (np.ndarray): grid for reaction coordinate
        xi (np.ndarray): trajectory of reaction coordinate
        la (np.ndarray): trajectory of fictitious particle
        sigma (float): thermal width of coupling of la to xi
        T (float): Temperature of the simulation

    returns:
        ti_force (np.ndarray): thermodynamic force (gradient of PMF)
    """
    RT = R_in_SI * T / 1000.0

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


def integrate(
    grad: np.ndarray, dx: float, T: float = 300.0, method: str = "trapezoid"
) -> Tuple[np.ndarray, np.ndarray]:
    """numeric integration of thermodynamic force by simpson, trapezoid or rectangle rule

    args:
        data (np.ndarray): data to integrate
        dx (float): bin width
        T (float): equillibrium temperature
        method (str): use 'simpson', 'trapezoid' or 'rectangle' rule

    returns:
        pmf (np.ndarray): potential of mean force
        rho (np.ndarray): probability density
    """
    RT = R_in_SI * T / 1000.0
    data = np.copy(grad)

    if method == "simpson":
        data[1:-1] *= 2.0
        data[2:-1:2] *= 2.0
        A = np.full(len(data), dx / 3.0)
    elif method == "trapezoid":
        data[1:-1] *= 2.0
        A = np.full(len(data), dx / 2.0)
    else:  # method == 'rectangle':
        A = np.full(len(data), dx)

    for i in range(len(A)):
        A[i] *= data[0 : i + 1].sum()

    rho = np.exp(-A / RT)
    rho /= rho.sum() * dx

    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))
    return pmf, rho


######################################################################
# Multistate Bennett Acceptance Ratio (Shirts et al. 2008)
######################################################################
def MBAR(
    traj_list: List[np.ndarray],
    meta_f: np.ndarray,
    max_iter: int = 10000,
    conv: float = 1.0e-7,
    outfreq: int = 100,
    T: float = 300.0,
    dV_list: List[np.ndarray] = None,
) -> np.ndarray:
    """self-consistent MBAR algorithm

    args:
        traj_list (list): List of biased trajectories
        meta_f (np.ndarray): input from metafile
        max_iter (int): Maximum number of iterations
        conv (float): Convergence criterion
        outfreq (int): Output frequency during self-consistent iteration
        T (float): Temperature of simulation
        dV_list (list): optional, list of GaMD boost potentials (has to match frames of traj_list)

    returns:
        W: array of statistical weigths of each frame
    """
    RT = R_in_SI * T / 1000.0
    beta = 1.0 / RT

    num_trajs = len(meta_f)

    print("Making Boltzmann factors\n")
    all_frames, num_frames, frames_per_traj = join_frames(traj_list)
    if dV_list is not None:
        all_dV, dV_num, dV_per_traj = join_frames(dV_list)
        all_dV *= atomic_to_kJmol  # kJ/mol
        if (dV_num != num_frames) or (frames_per_traj != dV_per_traj).all():
            raise ValueError("GaMD frames have to match eABF frames!")

    exp_U = []
    for _, line in enumerate(meta_f):
        if dV_list:
            exp_U.append(
                np.exp(
                    -beta
                    * (all_dV + 0.5 * line[2] * np.power(all_frames - line[1], 2)),
                    dtype=float,
                )
            )
        else:
            exp_U.append(
                np.exp(
                    -beta * 0.5 * line[2] * np.power(all_frames - line[1], 2),
                    dtype=float,
                )
            )

    exp_U = np.asarray(exp_U, dtype=float)  # this is a num_trajs x num_frames array

    beta_Ai = np.zeros(shape=(num_trajs,), dtype=float)

    print("All ready!\n")
    print("Start of the self-consistent iteration.")
    print("========================================================================")
    sys.stdout.flush()
    count = 0
    while True:
        count += 1

        denominator = np.multiply(frames_per_traj * np.exp(beta_Ai), exp_U.T)
        denominator = 1.0 / np.sum(denominator, axis=1)

        beta_Ai_new = -np.log(np.multiply(exp_U, denominator).sum(axis=1))
        beta_Ai_new -= beta_Ai_new[0]

        delta_Ai = np.abs(beta_Ai - beta_Ai_new)
        beta_Ai = np.copy(beta_Ai_new)

        if count % outfreq == 0 or count == 1:
            print("Iter %4d:\tConv=%14.10f" % (count, np.max(delta_Ai[1:])))
            sys.stdout.flush()

        if count == max_iter:
            print(
                "========================================================================"
            )
            print("Convergence not reached in {} iterations!".format(count))
            print(
                "Max error vector:", (_error_vec(frames_per_traj, beta_Ai, exp_U).max())
            )
            print(
                "========================================================================"
            )
            sys.stdout.flush()
            break

        if delta_Ai[1:].max() < conv:
            print(
                "========================================================================"
            )
            print("Converged after {} iterations!".format(count))
            print(
                "Max error vector:", (_error_vec(frames_per_traj, beta_Ai, exp_U).max())
            )
            print(
                "========================================================================"
            )
            sys.stdout.flush()
            break

    # final values
    weights = np.multiply(frames_per_traj * np.exp(beta_Ai), exp_U.T)
    weights = 1.0 / np.sum(weights, axis=1)

    return weights


def _error_vec(
    n_frames: np.ndarray, beta_Ai: np.ndarray, exp_U: np.ndarray
) -> np.ndarray:
    """error vector for MBAR"""
    denominator = np.multiply(n_frames * np.exp(beta_Ai), exp_U.T)
    denominator = 1.0 / np.sum(denominator, axis=1)
    # sum over all frames
    error_v = n_frames - n_frames * np.exp(beta_Ai) * (
        np.multiply(exp_U, denominator).sum(axis=1)
    )
    return error_v


def mbar_pmf(
    grid: np.array, cv: np.array, weights: np.array, T: float = 300.0
) -> Tuple[np.array, np.array]:
    """make free energy surface from MBAR result

    args:
        grid: grid along cv
        cv: trajectory of cv
        weights: boltzmann weights of frames in trajectory
        T: Temperature of simulation

    returns:
        Potential of mean force (PMF), probability density
    """
    RT = R_in_SI * T / 1000.0

    dx = grid[1] - grid[0]
    dx2 = dx / 2.0

    rho = np.zeros(len(grid), dtype=float)
    for ii, x in enumerate(grid):
        W_x = weights[np.where(np.logical_and(cv >= x - dx2, cv < x + dx2))]
        rho[ii] = W_x.sum()

    rho /= rho.sum() * dx
    pmf = -RT * np.log(rho, out=np.zeros_like(rho), where=(rho != 0))

    return pmf, rho


def mbar_deltapmf(
    TS: float, cv: np.array, weights: np.array, T: float = 300.0
) -> Tuple[np.array, np.array]:
    """Compute free energy difference from MBAR

    args:
        TS: position of transition state
        cv: trajectory of CV
        weights: MBAR weights of data points
        T: temperature

    returns:
        deltaA: free energy difference
    """
    RT = R_in_SI * T / 1000.0
    weights /= weights.sum()

    p_a = weights[np.where(cv < TS)].sum()
    p_b = weights[np.where(cv > TS)].sum()

    return -RT * np.log(p_b / p_a)


def ensemble_average(obs: np.ndarray, weights: np.ndarray) -> tuple:
    """ensemble average of observable

    args:
        obs (np.ndarray): trajectory of observables
        weights (np.ndarray): weigths of data frames

    returns:
        avg (float): ensemble average
        sem (float): standard error of the mean of avg
    """
    weights /= weights.sum()
    avg = np.average(obs, weights=weights)  # ensemble average
    std = np.sqrt(
        np.average(np.power(obs - avg, 2), weights=weights)
    )  # standard deviation
    sem = std / np.sqrt(float(len(obs)))  # standard error of the mean
    return avg, sem


def conditional_average(
    grid: np.ndarray, xi_traj: np.ndarray, obs: np.ndarray, weights: np.ndarray
) -> tuple:
    """bin wise conditional average and standard error of the mean

    args:
        grid (np.ndarray): bins along reaction coordinate
        xi_traj (np.ndarray): trajectory of reaction coordinate
        obs (np.ndarray): trajectory of observables
        weights (np.ndarray): weigths of data frames

    returns:
        obs_xi (np.ndarray): conditional average along reaction coordinate
        sem_xi (np.ndarray): standard error of the mean of obs_xi
    """
    weights /= weights.sum()
    dx = grid[1] - grid[0]
    dx2 = dx / 2

    obs_xi = np.zeros(len(grid))
    sem_xi = np.zeros(len(grid))

    for i, x in enumerate(grid):
        dat_i = np.where(np.logical_and(xi_traj >= x - dx2, xi_traj < x + dx2))
        nsamples = len(obs[dat_i])
        if nsamples > 0:
            p_i = weights[dat_i] / weights[dat_i].sum()
            obs_xi[i] = np.sum(p_i * obs[dat_i])
            sem_xi[i] = np.sqrt(
                np.sum(p_i * np.square(obs[dat_i] - obs_xi[i]))
            ) / np.sqrt(nsamples)
    return obs_xi, sem_xi


######################################################################
# Gaussian Accerated MD (Miao et al. 2015)
######################################################################
def gamd_correction_n(
    grid: np.ndarray,
    cv: np.ndarray,
    delta_V: np.ndarray,
    korder: int = 2,
    T: float = 300.0,
) -> tuple:
    """Correction to PMF from GaMD using kth cumulant expansion

    args:
        grid (np.ndarray): grid for reaction coordinate
        cv (np.ndarray): trajectory of reaction coordinate
        delta_V (np.ndarray): trajectory of GaMD bias potential
        korder (int): order of cumulant expansion (maximal 4)
        T (float): equillibrium temperature

    returns:
        corr (np.ndarray): correction to PMF from GaMD
        hist (np.ndarray): histogram of cv on grid
    """
    try:
        from scipy.stats import kstat
    except:
        raise ImportError("Failed to import scipy.stats.kstat")

    if korder > 4:
        print(" >>> Warning: cumulant expansion only supportet up to 4th order")
        korder = 4

    beta = 1.0 / (R_in_SI * T / 1000.0)  # 1/ kJ/mol
    dV_kJ = delta_V * atomic_to_kJmol  # kJ/mol

    dx2 = (grid[1] - grid[0]) / 2.0
    hist = np.zeros(len(grid))
    corr = np.zeros(len(grid))
    for i, x in enumerate(grid):
        dat_i = dV_kJ[np.where(np.logical_and(cv >= x - dx2, cv < x + dx2))]
        hist[i] = len(dat_i)
        if hist[i] > 0:
            # cumulant expansion to kth order
            for k in range(1, korder + 1):
                corr[i] += (np.power(beta, k) / math.factorial(k)) * kstat(dat_i, k)

    return -corr / beta, hist


def gamd_pmf(
    grid: np.ndarray,
    cv: np.ndarray,
    delta_V: np.ndarray,
    korder: int = 2,
    T: float = 300.0,
) -> tuple:
    """compute pmf from GaMD simulation

    args:
        grid (np.ndarray): grid for reaction coordinate
        cv (np.ndarray): trajectory of reaction coordinate
        delta_V (np.ndarray): trajectory of GaMD bias potential
        korder (int): order of cumulant expansion
        T (float): equilibrium temperature of simulations

    returns:
        pmf (np.ndarray): potential of mean force
        rho (np.ndarray): probability density
    """
    RT = R_in_SI * T / 1000.0  # kJ/mol
    dx = grid[1] - grid[0]

    # compute pmf from biased prob. density corrected by second order cumulant expansion
    gamd_corr, hist = gamd_correction_n(grid, cv, delta_V, korder=korder, T=T)
    rho = hist / (hist.sum() * dx)
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0)) + gamd_corr

    # get final results by normalizing prob. density
    rho = np.exp(-pmf / RT)
    rho /= rho[~np.isnan(pmf)].sum() * dx
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))

    return pmf, rho


######################################################################
# Gaussian Accelerated WTM-eABF (Chen et al. 2021)
######################################################################
def gaeabf_pmf(
    grid: np.ndarray,
    cv: np.ndarray,
    la: np.ndarray,
    sigma: float,
    delta_V: np.ndarray,
    korder: int = 2,
    T: float = 300.0,
) -> tuple:
    """compute pmf from GaWTM-eABF simulation

    args:
        grid: grid for reaction coordinate
        cv: trajectory of reaction coordinate
        la: trajectory of extended variable
        sigma: thermal width of coupling
        delta_V: trajectory of GaMD bias potential
        korder: order of cumulant expansion
        T: temperature

    returns:
        pmf (np.ndarray): potential of mean force
        rho (np.ndarray): probability density
    """
    RT = R_in_SI * T / 1000.0  # kJ/mol
    dx = grid[1] - grid[0]

    # compute pmf from CZAR and cumulant expansion
    czar = CZAR(grid, cv, la, sigma, T=T)
    pmf, _ = integrate(czar, dx, T=T)

    corr, _ = gamd_correction_n(grid, cv, delta_V, korder=korder, T=T)
    pmf += corr

    # get final results by normalizing prob. density
    rho = np.exp(-pmf / RT)
    rho /= rho[~np.isnan(pmf)].sum() * dx
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.nan), where=(rho != 0))

    return pmf, rho
