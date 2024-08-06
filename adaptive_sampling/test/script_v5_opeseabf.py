# ------------------------------------------------------------------------------------
# Script for OPESeABF 1d double well simulations
# ------------------------------------------------------------------------------------
import numpy as np

config_nsteps = 5e6  # Number of simulation steps
config_explore = False  # Enable Exploration mode
config_adaptive_sigma = False  # Enable adaptive sigma calculation according to welford
config_unbiased_time = 10  # Determine how many update freq steps an unbiased simulation is run to approximate sigma 0
config_input = True  # If False enable unbiased simulation to estimate sigma 0
config_fixed_sigma = (
    False  # Disable bandwidth rescaling and use input std for all kernels
)
config_merge = 1.0  # Merging or not, np.inf is no merging
config_update_freq = 200  # Frequency in which kernels are placed
config_recursion_merge = False  # Enable recursive merging
config_approx_norm_factor = True  # Enable approximation of norm factor
config_f_conf = 1000.0  # Confinment force to keep system in boundaries
config_bias_factor = None  # Direct setting of bias factor, if calculation from energy is wanted put in 'None'
config_energy_barr = 20.5  # Energy barrier to overcome
config_verbose = False  # Enable for debugging
config_enable_eabf = False  # Enable eABF
config_enable_opes = True  # Enable OPES
# ------------------------------------------------------------------------------------
import os
from sys import stdout

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

coords_in = [71.0, 0.5]

the_md = MD(
    mass_in=mass,
    coords_in=coords_in,
    potential="1",
    dt_in=dt,
    target_temp_in=temp,
    seed_in=seed,
)
the_md.calc_init()
the_md.calc_etvp()

# --------------------------------------------------------------------------------------
# define collective variables
cv_atoms = []  # not needed for 2D potentials
minimum = 70.0  # minimum of the CV
maximum = 170.0  # maximum of the CV
bin_width = 2.0  # bin with along the CV

collective_var = [["x", cv_atoms, minimum, maximum, bin_width]]

periodicity = np.array([None])
# ------------------------------------------------------------------------------------
# Setup the sampling algorithm
eabf_ext_sigma = 2.0  # thermal width of coupling between CV and extended variable
eabf_ext_mass = 20.0  # mass of extended variable in a.u.
abf_nfull = 500  # number of samples per bin when abf force is fully applied
kernel_std = np.array([4.89]) if config_input else np.array([None])

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


def print_output(the_md, the_bias):
    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14d\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_bias.ext_coords,
            the_md.epot,
            the_md.ekin,
            the_md.temp,
            len(the_bias.kernel_center),
            round(the_bias.potential, 5),
        )
    )
    stdout.flush()


# ------------------------------------------------------------------------------------
# Run MD
nsteps = config_nsteps
outfreq = 1000
x, y = [], []

print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % (
        "time [fs]",
        "x",
        "y",
        "lamda",
        "E_pot",
        "E_kin",
        "Temp",
        "# kernels",
        "bias pot",
    )
)
print_output(the_md, the_bias)

while the_md.step < nsteps:
    the_md.step += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    the_md.forces += the_bias.step_bias(write_output=False)

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    if the_md.step % outfreq == 0:
        print_output(the_md, the_bias)
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])
