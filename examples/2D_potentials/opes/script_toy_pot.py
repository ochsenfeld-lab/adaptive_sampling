# -------------------------------------------------------------------------------------
### Script for OPES test runs in predefined toy potentials
# -------------------------------------------------------------------------------------
import numpy as np
import os, sys, time
from adaptive_sampling.units import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.sampling_tools.opes import OPES
from adaptive_sampling.sampling_tools.opeseabf import OPESeABF

# ------------------------------------------------------------------------------------
# Parameters

nsteps = 5e6  # number of steps
update_freq = 200  # update frequency tau_G
explore = False  # enable explore mode
adaptive_std = True  # enable adaptive sigma rescaling
energy_barrier = 30.0  # approximated energy barrier
merge_threshold: float = np.inf  # merging threshold, np.inf disables merging
recursive_merge = True  # enable recursive merging
approximate_norm: bool = True  # linear scaling norm factor approximation

# ------------------------------------------------------------------------------------
# Setup CV

cv_atoms = []  # not needed for 2D potentials
min_1 = -60.0  # minimum of CV 1
max_1 = 60.0  # maximum of CV 1
bin_width_1 = 2.0  # bin width along CV 1
min_2 = -40.0  # minimum of CV 2
max_2 = 40.0  # minimum of CV 2
bin_width_2 = 2.0  # bin width along CV 2

collective_var = [
    ["x", cv_atoms, min_1, max_1, bin_width_1],
    #    ["y", cv_atoms, min_2, max_2, bin_width_2],
]

periodicity = None

grid_1 = np.arange(min_1, max_1, bin_width_1)
grid_2 = np.arange(min_2, max_2, bin_width_2)

# ------------------------------------------------------------------------------------
# Setup MD

mass = 10.0  # mass of particle in a.u.
seed = 42  # random seed
dt = 5.0e0  # stepsize in fs
temp = 300.0  # temperature in K
coords_in = [-50.0, 0.0]


the_md = MD(
    mass_in=mass,
    coords_in=coords_in,
    potential="2",
    dt_in=dt,
    target_temp_in=temp,
    seed_in=seed,
)
the_md.calc_init()
the_md.calc_etvp()

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
# --------------------------------------------------------------------------------------
output_freq = 1000  # frequency of writing outputs
kernel_std = np.array([None])  # std of initial kernel, if None it will be estimated

the_bias = OPES(
    the_md,
    collective_var,
    kernel_std=kernel_std,
    adaptive_std=adaptive_std,
    adaptive_std_freq=10,
    explore=explore,
    periodicity=periodicity,
    output_freq=output_freq,
    equil_temp=temp,
    energy_barr=energy_barrier,
    merge_threshold=merge_threshold,
    bias_factor=None,
    approximate_norm=approximate_norm,
    f_conf=100.0,
    numerical_forces=False,
    verbose=False,
)

# remove old output
if True:
    os.system("rm CV_traj.dat")
    os.system("rm restart_opes.npz")
    os.system("rm pmf_hist.npz")

the_bias.step_bias()


def print_output(the_md, the_bias, t):
    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14d\t%14.6f\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_bias.n_eff,
            the_md.temp,
            len(the_bias.kernel_center),
            the_bias.potential * kJ_to_kcal,
            t,
            the_bias.kernel_std[-1][0] if len(the_bias.kernel_std) > 0 else 0.0,
        )
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# Run MD
# --------------------------------------------------------------------------------------
traj_freq = 10
x, y, kernel_number, potentials = [], [], [], []
biased = True
pmf_hist = True

print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % (
        "time [fs]",
        "x",
        "y",
        "E_pot",
        "E_kin",
        "N_eff",
        "Temp",
        "n Kernel",
        "Bias Potential",
        "Wall time",
        "last sigma",
    )
)
print_output(the_md, the_bias, 0)

while the_md.step < nsteps:
    the_md.step += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    t0 = time.perf_counter()
    if biased:
        the_md.forces += the_bias.step_bias()
        potentials.append(the_bias.potential)
        if the_md.step % the_bias.update_freq == 0:
            kernel_number.append(len(the_bias.kernel_center))

    t = time.perf_counter() - t0
    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    if the_md.step % output_freq == 0:
        print_output(the_md, the_bias, t)

    if the_md.step % traj_freq == 0:
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])

# Save full trajectory for alternative reweighting
if True:
    np.savez("full_traj.npz", x=x, y=y)


# weighted PMF history
if True:
    cv_traj = np.loadtxt("CV_traj.dat", skiprows=1)
    full_traj = np.load("full_traj.npz")
    cv_1 = np.array(full_traj["x"])
    cv_2 = np.array(full_traj["y"])
    cv_pot = np.array(cv_traj[:, 4])
    pmf_history_1d, scattered_time_1d = the_bias.pmf_history_1d(
        cv_1, cv_pot, grid_1, hist_res=50
    )
    pmf_history_2d, scattered_time_2d = the_bias.pmf_history_2d(
        cv_1, cv_2, cv_pot, grid_1, grid_2, hist_res=50
    )
    np.savez(
        "pmf_hist.npz",
        pmf_history_1d=pmf_history_1d,
        scattered_time_1d=scattered_time_1d,
        pmf_weight_history_2d=pmf_history_2d,
        scattered_time_2d=scattered_time_2d,
    )
