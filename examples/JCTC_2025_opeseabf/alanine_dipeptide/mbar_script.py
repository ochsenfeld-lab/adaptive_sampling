import numpy as np
import sys
from typing import List, Tuple, Union, Dict
from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.units import R_in_SI

def mbar_postprocessing_2d(
    ext_sigma: Union[float, np.ndarray],
    bin_width: float = 2.0,
    borders: List[float] = [-180.0, 180.0, -180.0, 180.0],
    periodicity: List[List[float]] = [[-180.0, 180.0], [-180.0, 180.0]],
    traj_reduction: int = 1,
    equil_temp: float = 300.0,
    max_iter: int = 20000,
    conv: float = 1.0e-4,
    mbar_file: str = 'mbar_data.npz',
    cv_traj_file: str = 'CV_traj.dat',
    ):
    print("========================================================================")
    print("starting MBAR postprocessing...")
    cv_traj = np.loadtxt(cv_traj_file, skiprows=1)
    cv_x = np.array(cv_traj[:,1])[::traj_reduction]
    cv_y = np.array(cv_traj[:,2])[::traj_reduction]
    cv_la_x = np.array(cv_traj[:,3])[::traj_reduction]
    cv_la_y = np.array(cv_traj[:,4])[::traj_reduction]
    print(" > CV traj loaded")
    mbar_grid_x = np.arange(borders[0], borders[1], bin_width)
    mbar_grid_y = np.arange(borders[2], borders[3], bin_width)
    mbar_xx, mbar_yy = np.meshgrid(mbar_grid_x, mbar_grid_y)
    mbar_xy = np.array(list(zip(mbar_xx.ravel(),mbar_yy.ravel())))
    mbar_cv = np.array(list(zip(cv_x,cv_y)))
    mbar_la = np.array(list(zip(cv_la_x,cv_la_y)))
    print(" > meshgrid created")
    print("-----------------------------------")
    print(" Getting windows...")
    sys.stdout.flush()
    traj_list, indices, meta_f = mbar.get_windows(mbar_xy, mbar_cv, mbar_la, ext_sigma, equil_temp=equil_temp, dx=np.array([bin_width,bin_width]))
    print("-----------------------------------")
    print(" Building Boltzmann...")
    sys.stdout.flush()
    exp_U, frames_per_traj = mbar.build_boltzmann(
        traj_list,
        meta_f,
        equil_temp=equil_temp,
        periodicity=periodicity
    )
    print(" - Max Boltzmann is ", exp_U.max())
    print(" - Min Boltzmann is ", exp_U.min())
    print("-----------------------------------")
    print(" Initializing MBAR self consistent iteration...")
    sys.stdout.flush()
    weights = mbar.run_mbar(
        exp_U,
        frames_per_traj,
        max_iter=max_iter,
        conv=conv,
        conv_errvec=1.0,
        outfreq=100,
        device='cpu',
    )
    print(" >>> MBAR finished")
    print("========================================================================")
    print(" Calculating PMF from MBAR...")
    sys.stdout.flush()
    mbar_cv = mbar_cv[indices]
    RT = R_in_SI * equil_temp / 1000.0
    dx = np.array([bin_width,bin_width])
    dx2 = dx/2
    rho = np.zeros(shape=(len(mbar_xy),), dtype=float)
    for ii, center in enumerate(mbar_xy):
        indices = np.where(np.logical_and((mbar_cv >= center - dx2).all(axis=-1),
                                (mbar_cv < center + dx2).all(axis=-1)))
        rho[ii] = weights[indices].sum()
    rho /= rho.sum() * np.prod(dx)
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.NaN), where=(rho != 0))
    pmf = np.ma.masked_array(pmf, mask=np.isnan(pmf))
    pmf_mbar = pmf.reshape(mbar_xx.shape)
    rho_mbar = rho
    print(" >>> PMF aquired. Done!")

    np.savez(mbar_file, weights = weights, indices = indices, pmf_mbar = pmf_mbar, rho_mbar = rho_mbar)

    print(" - Saved!")


def mbar_direct_2d(
    mbar_cv: np.ndarray,
    mbar_la: np.ndarray,
    ext_sigma: Union[float, np.ndarray],
    bin_width: float = 2.0,
    borders: List[float] = [-180.0, 180.0, -180.0, 180.0],
    periodicity: List[List[float]] = [[-180.0, 180.0], [-180.0, 180.0]],
    equil_temp: float = 300.0,
    max_iter: int = 20000,
    conv: float = 1.0e-4,
    mbar_file: str = 'mbar_data.npz',
    ):
    print("========================================================================")
    print("starting MBAR postprocessing...")
    mbar_grid_x = np.arange(borders[0], borders[1], bin_width)
    mbar_grid_y = np.arange(borders[2], borders[3], bin_width)
    mbar_xx, mbar_yy = np.meshgrid(mbar_grid_x, mbar_grid_y)
    mbar_xy = np.array(list(zip(mbar_xx.ravel(),mbar_yy.ravel())))
    print(" > meshgrid created")
    print("-----------------------------------")
    print(" Getting windows...")
    sys.stdout.flush()
    traj_list, indices, meta_f = mbar.get_windows(mbar_xy, mbar_cv, mbar_la, ext_sigma, equil_temp=equil_temp, dx=np.array([bin_width,bin_width]))
    print("-----------------------------------")
    print(" Building Boltzmann...")
    sys.stdout.flush()
    exp_U, frames_per_traj = mbar.build_boltzmann(
        traj_list,
        meta_f,
        equil_temp=equil_temp,
        periodicity=periodicity
    )
    print(" - Max Boltzmann is ", exp_U.max())
    print(" - Min Boltzmann is ", exp_U.min())
    print("-----------------------------------")
    print(" Initializing MBAR self consistent iteration...")
    sys.stdout.flush()
    weights = mbar.run_mbar(
        exp_U,
        frames_per_traj,
        max_iter=max_iter,
        conv=conv,
        conv_errvec=1.0,
        outfreq=100,
        device='cpu',
    )
    print(" >>> MBAR finished")
    print("========================================================================")
    print(" Calculating PMF from MBAR...")
    sys.stdout.flush()
    mbar_cv = mbar_cv[indices]
    RT = R_in_SI * equil_temp / 1000.0
    dx = np.array([bin_width,bin_width])
    dx2 = dx/2
    rho = np.zeros(shape=(len(mbar_xy),), dtype=float)
    for ii, center in enumerate(mbar_xy):
        indices = np.where(np.logical_and((mbar_cv >= center - dx2).all(axis=-1),
                                (mbar_cv < center + dx2).all(axis=-1)))
        rho[ii] = weights[indices].sum()
    rho /= rho.sum() * np.prod(dx)
    pmf = -RT * np.log(rho, out=np.full_like(rho, np.NaN), where=(rho != 0))
    pmf = np.ma.masked_array(pmf, mask=np.isnan(pmf))
    pmf_mbar = pmf.reshape(mbar_xx.shape)
    rho_mbar = rho
    print(" >>> PMF aquired. Done!")

    np.savez(mbar_file, weights = weights, indices = indices, pmf_mbar = pmf_mbar, rho_mbar = rho_mbar)

    print(" - Saved!")