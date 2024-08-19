# ------------------------------------------------------------------------------------
# Script for OPESeABF 1d double well simulations
# ------------------------------------------------------------------------------------
import numpy as np

config_nsteps = 1e7  # Number of simulation steps
config_explore = False  # Enable Exploration mode
config_adaptive_sigma = False  # Enable adaptive sigma calculation according to welford
config_unbiased_time = 100  # Determine how many update freq steps an unbiased simulation is run to approximate sigma 0
config_input = True  # If False enable unbiased simulation to estimate sigma 0
config_fixed_sigma = (
    False  # Disable bandwidth rescaling and use input std for all kernels
)
config_merge = 1.0  # Merging or not, np.inf is no merging
config_update_freq = 200  # Frequency in which kernels are placed
config_recursion_merge = True  # Enable recursive merging
config_approx_norm_factor = True  # Enable approximation of norm factor
config_exact_norm_factor = (
    True  # Enable exact norm factor, if both activated, exact is used every 100 updates
)
config_f_conf = 1000.0  # Confinment force to keep system in boundaries
config_bias_factor = None  # Direct setting of bias factor, if calculation from energy is wanted put in 'None'
config_energy_barr = 50.0  # Energy barrier to overcome
config_verbose = False  # Enable for debugging
config_enable_eabf = True  # Enable eABF
config_enable_opes = True  # Enable eOPES
config_traj_reduction = 20  # Reduce length of trajectories to compute MBAR
# ------------------------------------------------------------------------------------
import os
from sys import stdout
import sys
import time
import nglview as ngl
import matplotlib.pyplot as plt
from adaptive_sampling.sampling_tools.opeseabf import OPESeABF
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *

# ------------------------------------------------------------------------------------
# setup MD
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
# define collective variables
cv_atoms = []  # not needed for 2D potentials
minimum = -60.0  # minimum of the CV
maximum = 60.0  # maximum of the CV
bin_width = 2.0  # bin with along the CV
min_y = -40.0
max_y = 40.0
bin_width_y = 2.0

collective_var = [
    ["x", cv_atoms, minimum, maximum, bin_width],
    ["y", cv_atoms, min_y, max_y, bin_width_y],
]

periodicity = None

# ------------------------------------------------------------------------------------
# Setup the sampling algorithm
eabf_ext_sigma = 1.0  # thermal width of coupling between CV and extended variable
eabf_ext_mass = 20.0  # mass of extended variable in a.u.
abf_nfull = 500  # number of samples per bin when abf force is fully applied
kernel_std = np.array([5.0, 2.0]) if config_input else np.array([None, None])

the_bias = OPESeABF(
    eabf_ext_sigma,
    eabf_ext_mass,
    the_md,
    collective_var,  # collective variable
    enable_eabf=config_enable_eabf,
    enable_opes=config_enable_opes,
    output_freq=1000,  # frequency of writing outputs
    nfull=abf_nfull,
    equil_temp=temp,  # equilibrium temperature of simulation
    force_from_grid=True,  # accumulate metadynamics force and bias on grid
    kinetics=True,  # calculate importent metrics to get accurate kinetics
    kernel_std=kernel_std,
    adaptive_sigma=config_adaptive_sigma,
    unbiased_time=config_unbiased_time,
    fixed_sigma=config_fixed_sigma,
    explore=config_explore,
    periodicity=periodicity,
    energy_barr=config_energy_barr,
    merge_threshold=config_merge,
    approximate_norm=config_approx_norm_factor,
    exact_norm=config_exact_norm_factor,
    verbose=config_verbose,
    recursion_merge=config_recursion_merge,
    update_freq=config_update_freq,
    f_conf=config_f_conf,
    bias_factor=config_bias_factor,
)
# ------------------------------------------------------------------------------------
if True:
    os.system("rm CV_traj.dat")

the_bias.step_bias(write_output=False)


def print_output(the_md, the_bias, t):
    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14d\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_bias.ext_coords[0],
            the_bias.ext_coords[1],
            the_md.epot,
            the_md.temp,
            len(the_bias.kernel_center),
            t,
            the_bias.potential,
        )
    )
    stdout.flush()


# ------------------------------------------------------------------------------------
# Run MD
nsteps = config_nsteps
outfreq = 1000
trajfreq = 10
x, y = [], []

print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % (
        "time [fs]",
        "x",
        "y",
        "lamda[0]",
        "lamda[1]",
        "E_pot",
        "Temp",
        "# kernels",
        "Wall time",
        "bias pot",
    )
)
print_output(the_md, the_bias, 0)

while the_md.step < nsteps:
    the_md.step += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    t0 = time.perf_counter()

    the_md.forces += the_bias.step_bias(write_output=False)

    t = time.perf_counter() - t0

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    if the_md.step % outfreq == 0:
        print_output(the_md, the_bias, t)

    if the_md.step % trajfreq == 0:
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])

# Save full trajectory for alternative reweighting
np.savez("full_traj.npz", x=x, y=y)

print("========================================================================")
print("### Simulation finished, start post processing... ###")
print("========================================================================")

from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.processing_tools.utils import autocorr
from adaptive_sampling.processing_tools.utils import ipce


cv_traj = np.loadtxt("CV_traj.dat", skiprows=1)
cv_x = np.array(cv_traj[:, 1])
cv_y = np.array(cv_traj[:, 2])
cv_la_x = np.array(cv_traj[:, 3])
cv_la_y = np.array(cv_traj[:, 4])

corr_x = autocorr(cv_x)
corr_y = autocorr(cv_y)

tau_x = ipce(corr_x)
tau_y = ipce(corr_y)

print("Autocorrelation times are ", tau_x, " for x and ", tau_y, " for y")

sys.stdout.flush()

traj_reduction = config_traj_reduction
cv_x = cv_x[::traj_reduction]
cv_y = cv_y[::traj_reduction]
cv_la_x = cv_la_x[::traj_reduction]
cv_la_y = cv_la_y[::traj_reduction]

ext_sigma = 1.0

# grid for free energy profile can be different than during sampling
minimum_x = -60.0
maximum_x = 60.0
minimum_y = -40.0
maximum_y = 40.0
bin_width = 1.0

# get tuple matrix of centers
mbar_grid_x = np.arange(minimum_x, maximum_x, bin_width)
mbar_grid_y = np.arange(minimum_y, maximum_y, bin_width)
mbar_xx, mbar_yy = np.meshgrid(mbar_grid_x, mbar_grid_y)
# mbar_xy = np.array(list(zip(mbar_xx.ravel(),mbar_yy.ravel())), dtype=('i4,i4')).reshape(mbar_xx.shape)
mbar_xy = np.array(list(zip(mbar_xx.ravel(), mbar_yy.ravel())))

# zip cv`s of both dimensions as well as extended coordinates
mbar_cv = np.array(list(zip(cv_x, cv_y)))
mbar_la = np.array(list(zip(cv_la_x, cv_la_y)))
# run MBAR and compute free energy profile and probability density from statistical weights
print("========================================================================")
print("Initialize MBAR...")
print("Getting windows...")

sys.stdout.flush()

traj_list, indices, meta_f = mbar.get_windows(
    mbar_xy, mbar_cv, mbar_la, ext_sigma, equil_temp=300.0, dx=np.array([1.0, 1.0])
)
print("========================================================================")
print("Build Boltzmann...")
sys.stdout.flush()

exp_U, frames_per_traj = mbar.build_boltzmann(
    traj_list,
    meta_f,
    equil_temp=300.0,
)

sys.stdout.flush()

print("========================================================================")
print("Initialize MBAR")
print("")
print("")
sys.stdout.flush()

weights = mbar.run_mbar(
    exp_U,
    frames_per_traj,
    max_iter=10000,
    conv=1.0e-4,
    conv_errvec=1.0,
    outfreq=100,
    device="cpu",
)

print("MBAR finished")
print("========================================================================")
print("")
print("")

print("Calculate PMF from MBAR")
sys.stdout.flush()

mbar_cv = mbar_cv[indices]
RT = R_in_SI * 300.0 / 1000.0

dx = np.array([1.0, 1.0])
dx2 = dx / 2
rho = np.zeros(shape=(len(mbar_xy),), dtype=float)
for ii, center in enumerate(mbar_xy):
    indices = np.where(
        np.logical_and(
            (mbar_cv >= center - dx2).all(axis=-1),
            (mbar_cv < center + dx2).all(axis=-1),
        )
    )
    rho[ii] = weights[indices].sum()

rho /= rho.sum() * np.prod(dx)
pmf = -RT * np.log(rho, out=np.full_like(rho, np.NaN), where=(rho != 0))
pmf = np.ma.masked_array(pmf, mask=np.isnan(pmf))
pmf_mbar = pmf.reshape(mbar_xx.shape)
rho_mbar = rho

print("PMF aquired, done!")

print("")
print("")
print("Postprocessing finished successfully.")

np.savez(
    "mbar_data.npz",
    weights=weights,
    indices=indices,
    pmf_mbar=pmf_mbar,
    rho_mbar=rho_mbar,
)

print("Saved!")
