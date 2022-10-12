import sys
import numpy as np
from typing import List, Tuple
from .utils import join_frames
from ..units import *


def run_mbar(
    exp_U: np.ndarray,
    frames_per_traj: np.ndarray, 
    max_iter: int = 10000,
    conv: float = 1.0e-7,
    conv_errvec: float = None,
    outfreq: int = 100,
) -> np.ndarray:
    """Self-consistent Multistate Bannett Acceptance Ratio (MBAR)

       see: Shirts et. al., J. Chem. Phys. (2008); https://doi.org/10.1063/1.2978177

    Args:
        exp_U: num_trajs*num_frames array of biased Boltzman factors
        frames_per_traj: number of samples per trajectory
        max_iter: Maximum number of iterations
        conv: Convergence criterion, largest change in beta*Ai
        conv_errvec: Convergence criterion based on the error vector, not used if None, largest absolute value in the error vec
        outfreq: Output frequency during self-consistent iteration

    Returns:
        W: array containing statistical weigths of each frame
    """
    num_trajs = len(frames_per_traj)
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

        max_err_vec = np.abs(_error_vec(frames_per_traj, beta_Ai, exp_U)).max()

        if count % outfreq == 0 or count == 1:
            print(
                "Iter %4d:\tConv=%14.10f\tConv_errvec=%14.6f"
                % (count, np.max(delta_Ai[1:]), max_err_vec)
            )
            sys.stdout.flush()

        if conv_errvec == None:
            converged = delta_Ai[1:].max() < conv
        else:
            converged = (delta_Ai[1:].max() < conv) and (max_err_vec < conv_errvec)

        if count == max_iter:
            print(
                "========================================================================"
            )
            print(f"Convergence not reached in {count} iterations!")
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
            print(f"Converged after {count} iterations!")
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

def build_boltzmann(
    traj_list: list, 
    meta_f: np.ndarray, 
    dU_list: list=None, 
    equil_temp: float=300.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build Boltzmann factores for MBAR
    
    Args:
        traj_list: list of trajectories
        meta_f: definition of simulation with force constants and cv grid
        dU_list: optional, additional potential that enters boltzmann factor (GaMD, confinement, reweighting, ...)
    
    Returns:
        exp_U: num_trajs**num_frames array of Boltzmann factors
        frames_per_traj: Number of frames per trajectory
    """
    RT = R_in_SI * equil_temp / 1000.0
    beta = 1.0 / RT
    
    all_frames, num_frames, frames_per_traj = join_frames(traj_list)
    if dU_list is not None:
        all_dU, dU_num, dU_per_traj = join_frames(dU_list)
        all_dU *= atomic_to_kJmol  # kJ/mol
        if (dU_num != num_frames) or (frames_per_traj != dU_per_traj).all():
            raise ValueError(" >>> Error: GaMD frames have to match eABF frames!")

    exp_U = []
    for _, line in enumerate(meta_f):
        if dU_list:
            exp_U.append(
                np.exp(
                    -beta
                    * (all_dU + 0.5 * line[2] * np.power(all_frames - line[1], 2)),
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
    return exp_U, frames_per_traj

def get_windows(
    centers: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: float,
    equil_temp: float = 300.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """generate mixture distribution of Gaussian shaped windows from eABF trajectory
       
       see: J. Chem. Phys. 157, 024110 (2022); https://doi.org/10.1063/5.0095554

    Args:
        centers: centers of windows
        xi: Trajectory of the reaction coordinate
        la: Trajectory of the extended variable
        sigma: Thermal width of coupling of xi and la
        equil_temp: equillibrium temperature

    Returns:
        traj_list: list of window trajectories,
        index_list: list of frame indices in original trajectory,
        meta_f: window information for MBAR (compatible with other popular MBAR/WHAM implementations)
    """
    RT = R_in_SI * equil_temp / 1000.0
    k = RT / (sigma * sigma)

    dx = centers[1] - centers[0]
    dx2 = dx / 2.0

    traj_list = []
    index_list = np.array([])
    for center in centers:
        indices = np.where(np.logical_and(la >= center - dx2, la < center + dx2))
        index_list = np.append(index_list, indices[0])
        traj_list += [xi[indices]]

    meta_f = np.zeros(shape=(len(centers), 3))
    meta_f[:, 1] = centers
    meta_f[:, 2] = k

    return traj_list, index_list.astype(np.int32), meta_f


def pmf_from_weights(
    grid: np.array, cv: np.array, weights: np.array, equil_temp: float = 300.0
) -> Tuple[np.array, np.array]:
    """make free energy surface from statistical weigths obtained by MBAR

    Args:
        grid: grid along cv
        cv: trajectory of cv
        weights: boltzmann weights of frames in trajectory
        equil_temp: Temperature of simulation

    Returns:
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
    """Compute free energy difference from statistical weigths obtained by MBAR

    Args:
        TS: position of transition state on CV
        cv: trajectory of CV
        weights: MBAR weights of data points
        equil_temp: temperature

    Returns:
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
