import numpy as np
from typing import List, Tuple, Union, Dict
from adaptive_sampling.units import *
from adaptive_sampling.processing_tools.mbar import pmf_from_weights
from adaptive_sampling.sampling_tools.utils import correct_periodicity

def pmf_from_reweighting(
    grid: np.ndarray,
    cv: np.ndarray,
    bias_pot: np.ndarray,
    equil_temp: float = 300.0,
    dx: np.ndarray = None,
    history_points: int = 1,
):
    """Get Potential of Mean Force (PMF) from statistical weights obtained by MBAR
       Note: for higher dimensional PMFs, they need to be reshaped,
       this function returns a flattened version

    Args:
        grid: centroids of grid along cv,
              shape=(N, d) for N centroids of d-dimension
        cv: trajectory of cv shape=(#frames, d)
        bias_pot: bias potentials for trajectory points
        equil_temp: Temperature of simulation
        dx: spacing between grid points
        history_points: number of history points to calculate

    Returns:
        pmf_hist: History of potential of mean force (PMF) in kJ/mol
        rho_hist: History of probability density
        scattered_time: Time points at which PMF was calculated
        history_points: Number of history points used

    """

    beta = 1. / (kB_in_atomic * equil_temp)
    W = np.exp(beta * np.asarray(bias_pot))

    if history_points < 1:
        history_points = 1
        print(" >>> INFO: At least one history point is required for reweighting... setting to 1.")
    
    scattered_time, pmf_hist, rho_hist = [], [], []

    if history_points == 1:
        pmf_weights, rho_weights = pmf_from_weights(grid, cv, W, equil_temp=equil_temp, dx=dx, verbose=True)
        pmf_weights -= pmf_weights.min()
        pmf_hist.append(pmf_weights)
        rho_hist.append(rho_weights)
    else:
        n = int(len(cv) / history_points)
        print_freq = int(history_points/5)
        for j in range(history_points):
            n_sample = j * n + n
            if j % print_freq == 0:
                print(f" >>> Progress: History entry {j} of {history_points}")
            scattered_time.append(n_sample)
            pmf_weights, rho_weights = pmf_from_weights(grid, cv[0:n_sample], W[0:n_sample], equil_temp=equil_temp, dx=dx, verbose=False)
            pmf_weights -= pmf_weights.min()
            pmf_hist.append(pmf_weights)
            rho_hist.append(rho_weights)

    return pmf_hist, rho_hist, scattered_time, history_points

def pmf_from_kernels(
    grid: np.ndarray,
    kernel_center: List,
    kernel_height: List,
    kernel_std: List,
    equil_temp: float = 300.0,
    energy_barrier: float = 20.0,
    explore: bool = False,
    periodicity: List = [None],
):
    """Get Potential of Mean Force (PMF) from superpositions of kernels with data from restart file

    Args:
        grid: centroids of grid along cv, for higher dimensions list of 1D grids
        kernel_center: centers of kernels
        kernel_height: heights of kernels
        kernel_std: standard deviations of kernels
        sum_weights: sum of weights of kernels
        n_iter: number of updates
        equil_temp: Temperature of simulation
        energy_barrier: energy barrier of simulation in kJ/mol
        explore: whether explore mode was used
        periodicity: periodicity of the system

    Returns:
        pmf_kernels: Potential of Mean Force (PMF) in kJ/mol
        probability_kernels: Probability Distribution
    """

    ncoords = len(kernel_center[0])
    #if ncoords != 1:
    #    raise ValueError(" >>> ERROR: Only 1D PMFs are supported for now...")
    n_kernel = len(kernel_center)
    beta = 1. / (kB_in_atomic * equil_temp)
    gamma = beta * (energy_barrier / atomic_to_kJmol)
    gamma_prefac = gamma - 1 if explore else 1 - 1 / gamma
    epsilon = np.exp((-beta * energy_barrier) / gamma_prefac)

    # Analytic calculation of norm factor
    sum_uprob = 0.0
    for s in kernel_center:
        if len(kernel_center) == 0:
            gaussians = 0.0
        else:
            s_diff = s - np.asarray(kernel_center)
            for i in range(ncoords):
                s_diff[:, i] = correct_periodicity(s_diff[:, i], periodicity[i])

            gaussians = np.asarray(kernel_height) * np.exp(
                -0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(kernel_std))), axis=1)
            )
        sum_uprob += np.sum(gaussians)
    norm_factor = sum_uprob / n_kernel

    # 1D
    if ncoords == 1:
        P = np.zeros_like(grid)
        for i in range(len(grid)):
            s_diff = grid[i] - np.asarray(kernel_center)
            for l in range(ncoords):
                s_diff[l] = correct_periodicity(s_diff[l], periodicity[l])
            val_gaussians = np.asarray(kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(kernel_std))),axis=1))
            P[i] = np.sum(val_gaussians)
        P /= P.sum()
        bias_pot = np.log(P/norm_factor + epsilon) / beta
        bias_pot = -gamma * bias_pot if explore else gamma_prefac * bias_pot 
        pmf_kernels = bias_pot/-gamma_prefac if not explore else bias_pot
        pmf_kernels -= pmf_kernels.min()
        probability_kernels = P/P.max()

        return pmf_kernels * atomic_to_kJmol, probability_kernels

    # 2D
    elif ncoords == 2:
        P = np.zeros([len(grid[0]), len(grid[1])])
        for i,x in enumerate(grid[0]):
            for j,y in enumerate(grid[1]):
                s_diff = np.array([x, y]) - np.asarray(kernel_center)
                for l in range(ncoords):
                    s_diff[:,l] = correct_periodicity(s_diff[:,l], periodicity[l])
                val_gaussians = np.asarray(kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(kernel_std))),axis=1))
                P[i,j] = np.sum(val_gaussians)
        bias_pot = np.log(P/norm_factor + epsilon) / beta
        bias_pot = -gamma * bias_pot if explore else gamma_prefac * bias_pot 
        pmf_kernels = bias_pot/-gamma_prefac if not explore else bias_pot
        pmf_kernels -= pmf_kernels.min()
        probability_kernels = P/P.max()

        return pmf_kernels * atomic_to_kJmol, probability_kernels
    
    else:
        raise ValueError(" >>> ERROR: Only 1D and 2D PMFs are supported for now...")