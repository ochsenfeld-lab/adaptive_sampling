import sys
import numpy as np
from typing import List, Tuple, Union, Dict
from .utils import join_frames
from ..units import *
import torch


def run_mbar(
    exp_U: np.ndarray,
    frames_per_traj: np.ndarray, 
    max_iter: int = 10000,
    conv: float = 1.0e-7,
    conv_errvec: float = None,
    outfreq: int = 100,
    device = 'cpu'
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

    frames_per_traj = torch.from_numpy(frames_per_traj.astype(np.float64))
    frames_per_traj = frames_per_traj.to(device=device)
    exp_U = torch.from_numpy(exp_U.astype(np.float64))
    exp_U = exp_U.to(device=device)

    num_trajs = len(frames_per_traj)
    beta_Ai = torch.zeros(size=(num_trajs,))
    beta_Ai = beta_Ai.to(device=device)
    #Ai_overtime = [beta_Ai]
    
    # First denominator with all zero Ai guess
    denominator = 1.0 / torch.matmul(frames_per_traj * torch.exp(beta_Ai), exp_U)
    expU_dot_denom = torch.matmul(exp_U, denominator)

    #print("All ready!\n")
    print("Start of the self-consistent iteration.")
    print("========================================================================")
    sys.stdout.flush()
    count = 0
    while True:
        count += 1

        beta_Ai_new = -torch.log(expU_dot_denom)
        beta_Ai_new -= torch.clone(beta_Ai_new[0])
        #Ai_overtime.append(beta_Ai_new)

        delta_Ai = torch.abs(beta_Ai - beta_Ai_new)
        beta_Ai = torch.clone(beta_Ai_new)
        
        prefac = frames_per_traj * torch.exp(beta_Ai)
        denominator = 1.0 / torch.matmul(prefac, exp_U)
        expU_dot_denom = torch.matmul(exp_U, denominator)

        error_v = (frames_per_traj - prefac * expU_dot_denom)
        max_err_vec = torch.abs(error_v).max()

        if count % outfreq == 0 or count == 1:
            print(
                "Iter %4d:\tConv=%14.10f\tConv_errvec=%14.6f"
                % (count, delta_Ai[1:].max(), max_err_vec)
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
            print(f"Max error vector: {max_err_vec:14.6f}")
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
            print(f"Max error vector: {max_err_vec:14.6f}")
            print(
                "========================================================================"
            )
            sys.stdout.flush()
            break

    # final values
    weights = 1.0 / torch.matmul(frames_per_traj * torch.exp(beta_Ai), exp_U)
    weights = weights.cpu().numpy()

    torch.cuda.empty_cache()

    return weights


def build_boltzmann(
    traj_list: List, 
    meta_f: np.ndarray, 
    dU_list: List=None,
    equil_temp: float=300.0,
    periodicity: Union[List, np.ndarray]=None,
    constraints: List[Dict]=None,
    progress_bar: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build Boltzmann factores for MBAR
    
    Args:
        traj_list: list of trajectories
        meta_f: definition of simulation with force constants and cv grid
        dU_list: optional, additional potential in a.u. that enters boltzmann factor (GaMD, confinement, reweighting, ...)
        equil_temp: equilibrium temperature of the simulation
        periodicity: list with upper and lower bound for periodic CVs such as angles or PBC boxes
        constraints: list of dictionaries for harmonic constraints, that were the same everywhere but necessary to constrain simulation
        progress_bar: show progress bar for generation of Boltzmann factors

    Returns:
        exp_U: num_trajs**num_frames array of Boltzmann factors
        frames_per_traj: Number of frames per trajectory
    """
    RT = R_in_SI * equil_temp / 1000.0
    beta = 1.0 / RT
    
    if periodicity:
        lower_bound = periodicity[0]
        upper_bound = periodicity[1]
        period = upper_bound - lower_bound

    all_frames, num_frames, frames_per_traj = join_frames(traj_list)

    # adding an extra axis to 1D sims, for compatibility with x-D eABF
    add_axis = True if periodicity and not hasattr(lower_bound, "__len__") else False
    if add_axis:
        all_frames = all_frames[:, np.newaxis]
        if periodicity:
            if type(lower_bound) == float:
                lower_bound = np.array([lower_bound])
                upper_bound = np.array([upper_bound])
                period = np.array([period])

    if dU_list is not None:
        all_dU, dU_num, dU_per_traj = join_frames(dU_list)
        all_dU *= atomic_to_kJmol  # kJ/mol
        if (dU_num != num_frames) or (frames_per_traj != dU_per_traj).all():
            raise ValueError(" >>> Error: Frames of external potential have to match eABF frames!")
        if add_axis:
            all_dU = all_dU[:, np.newaxis]

    if constraints:
        for const_dict in constraints:
            const_frames, _ , _ = join_frames(const_dict['traj_list'])
            diffs = const_frames - const_dict['eq_value']
            if 'period' in const_dict.keys():
                const_lb = const_dict['period'][0]
                const_ub = const_dict['period'][1]
                const_period = const_ub - const_lb
                diffs[diffs > const_ub] -= const_period
                diffs[diffs < const_lb] += const_period
            if const_energy:
                const_energy += 0.5 * const_dict['k'] * np.power(diffs, 2)
            else:
                const_energy = 0.5 * const_dict['k'] * np.power(diffs, 2)
            
    exp_U = []

    if progress_bar:
        from tqdm import tqdm
        meta_f = tqdm(meta_f)

    for line in meta_f:
        diffs = np.asarray(all_frames - line[1])
        if periodicity:
            for ii in range(diffs.shape[1]):
                diffs[diffs[:,ii] > upper_bound[ii], ii] -= period[ii]
                diffs[diffs[:,ii] < lower_bound[ii], ii] += period[ii]

        exp_U.append(
            np.exp(
                -beta * 0.5 * (line[2] * np.power(diffs, 2)).sum(axis=1),
                dtype=np.float64,
            )
        )    
        
        if dU_list:
            exp_U[-1] *= np.exp(-beta * all_dU.sum(axis=1))
            
        if constraints:
            exp_U[-1] *= np.exp(-beta * const_energy, dtype=np.float64)

    exp_U = np.asarray(exp_U, dtype=np.float64)   # this is a num_trajs x num_frames list
    return exp_U, frames_per_traj


def get_windows(
    centers: np.ndarray,
    xi: np.ndarray,
    la: np.ndarray,
    sigma: Union[float, np.ndarray],
    equil_temp: float = 300.0,
    dx: np.ndarray = None,
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
    k = RT / np.power(sigma, 2)

    if dx is None:
        dx = centers[1] - centers[0]
    dx2 = dx / 2.0
    
    # assumes if true for one then true for all
    if len(xi.shape) == 1:
        xi = xi[:, np.newaxis]
        la = la[:, np.newaxis]
        centers = centers[:, np.newaxis]
        
    traj_list = []
    index_list = np.array([])

    for center in centers:
        indices = np.where(np.logical_and((la >= center - dx2).all(axis=-1), 
                              (la < center + dx2).all(axis=-1)))
        index_list = np.append(index_list, indices[0])
        traj_list += [xi[indices]]

    meta_f = np.zeros(shape=(3, *centers.shape))
    meta_f[1] = centers
    meta_f[2] = k

    meta_f = meta_f.transpose((1,0,2))

    return traj_list, index_list.astype(np.int32), meta_f


def pmf_from_weights(
    grid: np.ndarray,
    cv: np.ndarray,
    weights: np.ndarray,
    equil_temp: float = 300.0,
    dx: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get Potential of Mean Force (PMF) from statistical weigths obtained by MBAR
       Note: for higher dimensional PMFs, they need to be reshaped, 
       this function returns a flattened version

    Args:
        grid: centroids of grid along cv, 
              shape=(N, d) for N centroids of d-dimension
        cv: trajectory of cv shape=(#frames, d)
        weights: boltzmann weights of frames in trajectory
        equil_temp: Temperature of simulation

    Returns:
        pmf: Potential of mean force (PMF) in kJ/mol
        rho: probability density
    """
    RT = R_in_SI * equil_temp / 1000.0

    if dx is None:
        dx = grid[1] - grid[0]
    dx2 = dx / 2.0

    if len(grid.shape) == 1:
        cv = cv[:, np.newaxis]
        grid = grid[:, np.newaxis]

    rho = np.zeros(shape=(len(grid),), dtype=float)
    for ii, center in enumerate(grid):
        indices = np.where(np.logical_and((cv >= center - dx2).all(axis=-1),
                              (cv < center + dx2).all(axis=-1)))
        rho[ii] = weights[indices].sum()


    rho /= rho.sum() * np.prod(dx)
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.NaN), where=(rho != 0))
    pmf = np.ma.masked_array(pmf, mask=np.isnan(pmf))

    return pmf, rho
