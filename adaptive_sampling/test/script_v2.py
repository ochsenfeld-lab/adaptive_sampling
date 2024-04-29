import numpy as np

# CONFIGURE PARAMETER
config_nsteps = 3e6
config_merge = 1.0          #Merging or not, np.inf is no merging
config_update_freq = 300
config_print_freq = 1000
config_approx_norm_factor = True
config_recursion_merge = False
config_verbose = False

#-------------------------------------------------------------------------------------
import os
from sys import stdout
import time
import nglview as ngl
import matplotlib.pyplot as plt
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *
import sys

def assymetric_potential(coord_x, coord_y):
    """Analytical asyymetric double well potential
    """
    a = 0.005
    b = 0.040
    d = 40.0
    e = 20.0

    exp_1 = np.exp((-a * (coord_x - d) * (coord_x - d)) + (-b * (coord_y - e) * (coord_y - e)))
    exp_2 = np.exp((-a * (coord_x + d) * (coord_x + d)) + (-b * (coord_y + e) * (coord_y + e)))

    return -np.log(exp_1 + exp_2) / atomic_to_kJmol

coords_x = np.arange(-60,60,1.0)
coords_y = np.arange(-40,40,1.0)
xx,yy = np.meshgrid(coords_x,coords_y)

PES = assymetric_potential(xx,yy)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# SETUP MD
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms        = []              # not needed for 2D potentials
minimum         = -60.0            # minimum of the CV
maximum         = 60.0           # maximum of the CV
bin_width       = 2.0             # bin with along the CV
min_y           = -40.0
max_y           = 40.0
bin_width_y     = 2.0

collective_var = [
    ["x", cv_atoms, minimum, maximum, bin_width],
    ["y", cv_atoms, min_y, max_y, bin_width_y]
]

# ------------------------------------------------------------------------------------
# setup MD
mass      = 10.0   # mass of particle in a.u.
seed      = 42     # random seed
dt        = 5.0e0  # stepsize in fs
temp      = 300.0  # temperature in K

coords_in = [-50.0, -30.0]

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
output_freq       = 1000    # frequency of writing outputs
kernel_std        = np.array([5.0,1.0])

the_bias = OPES(
    kernel_std,
    the_md,
    collective_var,
    output_freq=output_freq,
    equil_temp=temp,
    energy_barr = 20.0,
    merge_threshold=config_merge,
    approximate_norm=config_approx_norm_factor,
    verbose=config_verbose,
    recursion_merge=config_recursion_merge,
    update_freq = config_update_freq,
)

# --------------------------------------------------------------------------------------
# remove old output
if True:
    os.system("rm CV_traj.dat")
    os.system("rm restart_opes.npz")

the_bias.step_bias()

def print_output(the_md,the_bias, t):
    print("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14d\t%14.6f\t%14.6f" % (
        the_md.step * the_md.dt * atomic_to_fs,
        the_md.coords[0],
        the_md.coords[1],
        the_md.epot,
        the_md.ekin,
        the_md.epot + the_md.ekin,
        the_md.temp,
        len(the_bias.kernel_center),
        the_bias.potential,
        t,
    ))  
    sys.stdout.flush()

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# RUN MD
# --------------------------------------------------------------------------------------
nsteps     = config_nsteps
traj_freq  = 10
print_freq = 1000
x,y        = [],[]
biased     = True
kernel_number = []
potentials = []


print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % ("time [fs]", "x", "y", "E_pot", "E_kin", "E_tot", "Temp","n Kernel", "Bias Potential", "Wall time")
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
