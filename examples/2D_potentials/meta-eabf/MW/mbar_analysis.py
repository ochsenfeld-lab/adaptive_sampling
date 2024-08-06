import numpy as np
from adaptive_sampling.processing_tools import mbar

traj1 = np.loadtxt("W1/CV_traj1.dat", skiprows=1)
traj2 = np.loadtxt("W2/CV_traj1.dat", skiprows=1)

ext_sigma = 2.0

minimum = traj1.min()
maximum = traj1.max()
bin_width = 2.0

grid = np.arange(minimum, maximum, bin_width)

cv1 = traj1[:, 1]  # trajectory of collective variable
la1 = traj1[:, 2]  # trajectory of extended system (for US: center of window)

traj1_list, indices1, meta_f1 = mbar.get_windows(
    grid, cv1, la1, ext_sigma, equil_temp=300.0
)

exp_U1, frames_per_traj1 = mbar.build_boltzmann(
    traj1_list,
    meta_f1,
    equil_temp=300.0,
)

weights1 = mbar.run_mbar(
    exp_U1,
    frames_per_traj1,
    max_iter=10000,
    conv=1.0e-7,
    conv_errvec=1.0,
    outfreq=100,
    device="cpu",
)

pmf1, _ = mbar.pmf_from_weights(grid, cv1[indices1], weights1, equil_temp=300.0)

cv2 = traj2[:, 1]  # trajectory of collective variable
la2 = traj2[:, 2]  # trajectory of extended system (for US: center of window)

traj2_list, indices2, meta_f2 = mbar.get_windows(
    grid, cv2, la2, ext_sigma, equil_temp=300.0
)

exp_U2, frames_per_traj2 = mbar.build_boltzmann(
    traj2_list,
    meta_f2,
    equil_temp=300.0,
)

weights2 = mbar.run_mbar(
    exp_U2,
    frames_per_traj2,
    max_iter=10000,
    conv=1.0e-7,
    conv_errvec=1.0,
    outfreq=100,
    device="cpu",
)

pmf2, _ = mbar.pmf_from_weights(grid, cv2[indices1], weights2, equil_temp=300.0)

weights_full = np.append(weigths1, weights2)
weights_full /= weights_full.sum()

cv_full = np.append(cv1[indices1], cv2[indices2])

pmf_full, _ = mbar.pmf_from_weights(grid, cv_full, weights_full, equil_temp=300.0)

np.savez(
    "results",
    pmf1=pmf1,
    pmf2=pmf2,
    pmf=pmf_full,
)
