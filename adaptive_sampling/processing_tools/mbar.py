import os
import sys
import numpy as np
from typing import List, Tuple

from .utils import _join_frames
from ..units import *


def mbar(
    traj_list: List[np.ndarray],
    meta_f: np.ndarray,
    max_iter: int = 10000,
    conv: float = 1.0e-7,
    conv_errvec: float = None,
    outfreq: int = 100,
    equil_temp: float = 300.0,
    dV_list: List[np.ndarray] = None,
) -> np.ndarray:
    """self-consistent MBAR algorithm

    args:
        traj_list: List of biased trajectories
        meta_f: input from metafile
        max_iter: Maximum number of iterations
        conv: Convergence criterion, largest change in beta*Ai
        conv_errvec: Convergence criterion based on the error vector, not used if None, largest absolute value in the error vec
        outfreq: Output frequency during self-consistent iteration
        equil_temp: Temperature of simulation
        dV_list: optional, list of GaMD boost potentials (has to match frames of traj_list)

    returns:
        W: array of statistical weigths of each frame
    """
    RT = R_in_SI * equil_temp / 1000.0
    beta = 1.0 / RT

    num_trajs = len(meta_f)

    print("Making Boltzmann factors\n")
    all_frames, num_frames, frames_per_traj = _join_frames(traj_list)
    if dV_list is not None:
        all_dV, dV_num, dV_per_traj = _join_frames(dV_list)
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
    Ai_overtime = [beta_Ai]

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
        Ai_overtime.append(beta_Ai_new)

        delta_Ai = np.abs(beta_Ai - beta_Ai_new)
        beta_Ai = np.copy(beta_Ai_new)

        if count % outfreq == 0 or count == 1:
            print("Iter %4d:\tConv=%14.10f" % (count, np.max(delta_Ai[1:])))
            sys.stdout.flush()

        max_err_vec = np.abs(_error_vec(frames_per_traj, beta_Ai, exp_U)).max()
        if conv_errvec == None:
            converged = delta_Ai[1:].max() < conv
        else:
            converged = (delta_Ai[1:].max() < conv) and (max_err_vec < conv_errvec)

        if count == max_iter:
            print(
                "========================================================================"
            )
            print("Convergence not reached in {} iterations!".format(count))
            print("Max error vector:", max_err_vec)
            print(
                "========================================================================"
            )
            sys.stdout.flush()
            break

        if converged:
            print(
                "========================================================================"
            )
            print("Converged after {} iterations!".format(count))
            print("Max error vector:", max_err_vec)
            print(
                "========================================================================"
            )
            sys.stdout.flush()
            break

    # final values
    weights = np.multiply(frames_per_traj * np.exp(beta_Ai), exp_U.T)
    weights = 1.0 / np.sum(weights, axis=1)

    return weights


def get_windows(
    centers: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    equil_temp: float = 300.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """generate US windows from eABF trajectory

    args:
        centers (np.ndarray): centers of Umbrella windows
        xi (np.ndarray): Trajectory of the reaction coordinate
        la (np.ndarray): Trajectory of the extended variable
        sigma (float): Thermal width of coupling of xi and la
        equil_temp (float): equillibrium temperature

    returns:
        traj_list (list): list of window trajectories,
        index_list (np.ndarray): list of frame indices in original trajectory,
        meta_f (np.ndarray): window information for MBAR
    """
    RT = R_in_SI * equil_temp / 1000.0
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


def write_mbar_data(
    centers: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    E_pot=None,
    equil_temp: float = 300.0,
    out: str = "umbrella_data/",
):
    """Get subsampled trajectories and metafile from eABF data
    to use popular WHAM/MBAR/EMUS implementations with eABF
    """
    if not os.path.isdir(out):
        os.mkdir(out)
    os.chdir(out)

    traj_list, _, meta_f = get_windows(centers, xi, la, sigma, equil_temp=equil_temp)

    for i in range(len(centers)):
        out = open("cv_meta.dat", "a")
        out.write("umbrella_%d.dat\t%14.6f\t%14.6f\n" % (i, meta_f[i, 1], meta_f[i, 2]))
        out.close()

        timeseries = open("umbrella_%d.dat" % i, "w")
        for t, cv in enumerate(traj_list[i]):
            timeseries.write("%d\t%14.6f\n" % (t, cv))
        timeseries.close()

    if E_pot is not None:
        E_list, _, _ = get_windows(centers, E_pot, la, sigma, equil_temp=equil_temp)
        for i in range(len(centers)):
            e_series = open("epot_%d.dat" % i, "w")
            for t, epot in enumerate(E_list[i]):
                e_series.write("%d\t%14.6f\n" % (t, epot))
            e_series.close()


def pmf_from_weights(
    grid: np.array, cv: np.array, weights: np.array, equil_temp: float = 300.0
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
    RT = R_in_SI * equil_temp / 1000.0

    dx = grid[1] - grid[0]
    dx2 = dx / 2.0

    rho = np.zeros(len(grid), dtype=float)
    for ii, x in enumerate(grid):
        W_x = weights[np.where(np.logical_and(cv >= x - dx2, cv < x + dx2))]
        rho[ii] = W_x.sum()

    rho /= rho.sum() * dx
    pmf = -RT * np.log(rho, out=np.zeros_like(rho), where=(rho != 0))

    return pmf, rho


def deltaf_from_weights(
    TS: float, cv: np.array, weights: np.array, equil_temp: float = 300.0
) -> Tuple[np.array, np.array]:
    """Compute free energy difference from weights

    args:
        TS: position of transition state on CV
        cv: trajectory of CV
        weights: MBAR weights of data points
        T: temperature

    returns:
        deltaF: free energy difference
    """
    RT = R_in_SI * equil_temp / 1000.0
    weights /= weights.sum()

    p_a = weights[np.where(cv < TS)].sum()
    p_b = weights[np.where(cv > TS)].sum()

    return -RT * np.log(p_b / p_a)


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
