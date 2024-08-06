import numpy as np

# CONFIGURE PARAMETER
config_nsteps = 3e6
config_merge = 1.0  # Merging or not, np.inf is no merging
config_update_freq = 300
config_print_freq = 1000
config_approx_norm_factor = True
config_recursion_merge = False

# -------------------------------------------------------------------------------------
import os
from sys import stdout
import time
import nglview as ngl
import matplotlib.pyplot as plt
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *
import sys


def double_well_potential(coord_x, coord_y):
    """Analytical double well potential"""
    a = 8.0e-6
    b = 0.5
    d = 80.0
    e = 160.0

    s1 = (coord_x - d) * (coord_x - d)
    s2 = (coord_x - e) * (coord_x - e)

    return a * s1 * s2 + b * coord_y * coord_y


coords_x = np.arange(60, 181, 1.0)
coords_y = np.arange(-8, 8, 0.1)
xx, yy = np.meshgrid(coords_x, coords_y)

PES = double_well_potential(xx, yy)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# SETUP MD
# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms = []  # not needed for 2D potentials
minimum = 70.0  # minimum of the CV
maximum = 170.0  # maximum of the CV
bin_width = 2.0  # bin with along the CV

collective_var = [["x", cv_atoms, minimum, maximum, bin_width]]

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
# Setup the sampling algorithm
eabf_ext_sigma = 2.0  # thermal width of coupling between CV and extended variable
eabf_ext_mass = 50.0  # mass of extended variable in a.u.
abf_nfull = 500  # number of samples per bin when abf force is fully applied
mtd_hill_height = 1.0  # MtD hill height in kJ/mol
mtd_hill_std = 4.0  # MtD hill width
mtd_well_tempered = 1000.0  # MtD Well-tempered temperature
mtd_frequency = 100  # MtD frequency of hill creation
output_freq = 1000  # frequency of writing outputs
kernel_std = np.array([5.0])

the_bias = OPES(
    kernel_std,
    the_md,
    collective_var,
    output_freq=output_freq,
    equil_temp=temp,
    energy_barr=20.0,
    merge_threshold=config_merge,
    approximate_norm=config_approx_norm_factor,
    verbose=False,
    recursion_merge=config_recursion_merge,
    update_freq=config_update_freq,
)

# --------------------------------------------------------------------------------------
# remove old output
if True:
    os.system("rm CV_traj.dat")

the_bias.step_bias()


def print_output(the_md, the_bias, t):
    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14d\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_md.epot + the_md.ekin,
            the_md.temp,
            len(the_bias.kernel_center),
            round(the_bias.potential, 5),
            t,
        )
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# RUN MD
# --------------------------------------------------------------------------------------
nsteps = config_nsteps
traj_freq = 10
print_freq = 1000
x, y = [], []
biased = True
kernel_number = []
potentials = []


print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % (
        "time [fs]",
        "x",
        "y",
        "E_pot",
        "E_kin",
        "E_tot",
        "Temp",
        "n Kernel",
        "Bias Potential",
        "Wall time",
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

    if the_md.step % print_freq == 0:
        print_output(the_md, the_bias, t)

    if the_md.step % traj_freq == 0:
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])
