import numpy as np
from typing import Union, List, Tuple


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


def delta_pmf(
    pmf: np.ndarray, T: float = 300.0, min_bin: int = 20, max_bin: int = -20
) -> tuple:
    """calculate free energy difference

    args:
        pmf: potential of mean force (free energy surface)
        T: temperature
        min_bin/max_bin: minimum/maximum bin for search of transition state

    returns:
        dA (float): free energy difference
        dA_grid (np.ndarray): free energy difference on grid
    """
    R = 8.314 / 1000.0  # kJ / K mol
    RT = (R * T) / 1000.0
    pmf = pmf[~np.isnan(pmf)]

    TS = np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0]

    P = np.exp(-pmf / RT)
    P /= P.sum()

    P_a = P[:TS].sum()
    P_b = P[(TS + 1) :].sum()

    dA = RT * np.log(P_a / P_b)
    dA_grid = np.zeros(len(pmf))
    dA_grid[TS] += dA / 2
    dA_grid[(TS + 1) :] += dA

    return dA, dA_grid


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


def _join_frames(traj_list: List[np.array]) -> Tuple[np.ndarray, float, np.ndarray]:
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
