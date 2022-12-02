import numpy as np
from typing import Union, List, Tuple
from ..units import *


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


def _next_pow_two(n: int):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr(x: np.array) -> np.array:
    """Estimate the normalized autocorrelation function of a 1-D series

    args:
        x: The time series of which to calculate the autocorrelation function.

    returns:
        acfxn: The autocorrelation as a function of lag time.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = _next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def ipce(corr_x):
    """ The initial positive correlation time estimator for the autocorrelation time, as proposed by Geyer et al.

    args
        x: The time series of which to calculate the autocorrelation function.

    returns:
        tau: Estimate of the autocorrelation time.
        mean: Average value of x
        sigma: variance of x
    """
    lag_max = int(len(corr_x) / 2)
    i, t = 0, 0.0
    while i < 0.5 * lag_max:
        gamma = corr_x[2 * i] + corr_x[2 * i + 1]
        if gamma < 0.0:
            #  print("stop at ", 2*i)
            break
        else:
            t += gamma
        i += 1
    tau = 2 * t - 1
    return tau


def join_frames(traj_list: List[np.array]) -> Tuple[np.ndarray, float, np.ndarray]:
    """get one array with all frames from list of trajectories

    Args:
        traj_list: list of trajectories

    Returns:
        all_frames: array with all samples
        num_frames: total numper of samples
        frames_per_traj: array with number of samples per original trajectorie
    """
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


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm

    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample

    Returns:
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


def weighted_hist(grid: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """get weighted histogram of data

    Args:
        grid: bins along reaction coordinate
        obs: data frames of observable
        weights: weights of data frames

    Returns:
        hist: weighted histogram
    """
    weights /= weights.sum()
    hist = np.zeros(len(grid))
    dx = grid[1] - grid[0]
    idx = np.arange(0, len(obs), 1)
    for i, x in enumerate(grid):
        dat_x = idx[np.where(np.logical_and(obs >= x - dx / 2.0, obs <= x + dx / 2.0))]
        hist[i] = (obs[dat_x] * weights[dat_x]).sum()
    hist /= hist.sum() * dx
    return hist


def ensemble_average(obs: np.ndarray, weights: np.ndarray) -> tuple:
    """ensemble average of observable

    Args:
        obs (np.ndarray): trajectory of observables
        weights (np.ndarray): weigths of data frames

    Returns:
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
) -> Tuple[np.ndarray, np.ndarray]:
    """bin wise conditional average and standard error of the mean

    Args:
        grid: bins along reaction coordinate
        xi_traj: trajectory of reaction coordinate
        obs: trajectory of observables
        weights: weigths of data frames

    Returns:
        obs_xi: conditional average along reaction coordinate
        sem_xi: standard error of the mean of obs_xi
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


def reaction_freeE(
    pmf: np.ndarray,
    T: float = 300.0,
    min_bin: int = 20,
    max_bin: int = -20,
    TS: int = None,
) -> tuple:
    """calculate free energy difference
       see: Dietschreit et al., J. Chem. Phys. 156, 114105 (2022); https://doi.org/10.1063/5.0083423

    Args:
        pmf: potential of mean force (free energy surface)
        T: temperature
        min_bin/max_bin: minimum/maximum bin for search of transition state
        TS: alternatively, bin number of TS

    Returns:
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
    pmf: np.ndarray,
    m_xi_inv: np.ndarray,
    T: float = 300.0,
    min_bin: int = 20,
    max_bin: int = -20,
    TS: int = None,
) -> tuple:
    """calculate activation free energy
       see: Dietschreit et al., J. Chem. Phys., 157, 084113 (2022).; <https://aip.scitation.org/doi/10.1063/5.0102075>

    Args:
        pmf: potential of mean force (free energy surface)
        m_xi_inv: z-conditioned average of inverse mass associates with CV, expected units are xi^2/(au_mass * angstrom^2)
        T: temperature
        min_bin/max_bin: minimum/maximum bin for search of transition state
        TS: alternatively, bin number of TS

    Returns:
        dA (float): free energy difference
        dA_grid (np.ndarray): free energy difference on grid
    """
    RT = (R_in_SI * T) / 1000.0  # kJ/mol

    pmf = pmf[~np.isnan(pmf)]
    if TS == None:
        TS = np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0]

    rho = np.exp(-pmf / RT)
    P = rho / rho.sum()  # normalize so that P_a + P_b = 1.0

    lambda_xi = np.sqrt(
        (h_in_SI * h_in_SI * m_xi_inv[TS]) / (2.0 * np.pi * atomic_to_kg * kB_in_SI * T)
    )
    lambda_xi *= 1e10

    P_a = P[:TS].sum()
    P_b = P[(TS + 1) :].sum()

    dA_a2b = -RT * np.log((rho[TS] * lambda_xi) / P_a)
    dA_b2a = -RT * np.log((rho[TS] * lambda_xi) / P_b)
    dA_a2b_approx = pmf[TS] - pmf[:TS].min()
    dA_b2a_approx = pmf[TS] - pmf[TS:].min()

    return dA_a2b, dA_b2a, dA_a2b_approx, dA_b2a_approx
