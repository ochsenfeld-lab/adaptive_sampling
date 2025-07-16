import os
from sys import stdout
import numpy as np

from adaptive_sampling import units
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *

# ------------------------------------------------------------------------------------
# MD parameters
nsteps     = 10005000
traj_freq  = 10
print_freq = 1000
x,y        = [],[]
biased     = True

# remove old output
if os.path.isfile('CV_traj.dat'):
    print('Removing old trajectory')
    os.system("rm CV_traj.dat")

# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms        = []              # not needed for 2D potentials
minimum         = -1.2            # minimum of the CV
maximum         = 1.0             # maximum of the CV
bin_width       = 0.05            # bin with along the CV

collective_var = [["x", cv_atoms, minimum, maximum, bin_width]]

# ------------------------------------------------------------------------------------
# setup MD
mass      = 10.0   # mass of particle in a.u.
seed      = np.random.randint(1000) # random seed
dt        = 1.0e0  # stepsize in fs
temp      = 300.0  # temperature in K

coords_in = [np.random.normal(-0.4, 0.1), np.random.normal(1.55, 0.1)]
print(f"STARTING MD FROM {coords_in}")

the_md = MD(
    mass_in=mass,
    coords_in=coords_in,
    potential="3",
    dt_in=dt,
    target_temp_in=temp,
    seed_in=seed,
)
the_md.calc_init()
the_md.calc_etvp()

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
eabf_ext_sigma    = None    # thermal width of coupling between CV and extended variable 
eabf_ext_mass     = 20.0    # mass of extended variable in a.u.
adaptive_coupling_stride = 5000
adaptive_coupling_scaling = 0.5
abf_nfull         = 500     # number of samples per bin when abf force is fully applied
mtd_hill_height   = 1.0     # MtD hill height in kJ/mol   
mtd_hill_std      = 0.1     # MtD hill width
mtd_wtm_temp      = 4200.0  # MtD Bias factor gamma
mtd_frequency     = 500     # MtD frequency of hill creation
output_freq       = 1000    # frequency of writing outputs
apply_abf         = True

the_bias = WTMeABF(
    the_md, 
    collective_var,           # collective variable
    ext_sigma=eabf_ext_sigma, 
    ext_mass=eabf_ext_mass, 
    adaptive_coupling_stride=adaptive_coupling_stride,
    adaptive_coupling_scaling=adaptive_coupling_scaling,
    nfull=abf_nfull,      
    apply_abf=apply_abf,
    well_tempered_temp=mtd_wtm_temp,
    hill_height=mtd_hill_height,
    hill_std=mtd_hill_std,
    hill_drop_freq=mtd_frequency,
    output_freq=output_freq,       
    f_conf=10000.0,           # confinement force of CV at boundaries
    equil_temp=temp,          # equilibrium temperature of simulation
    force_from_grid=True,     # accumulate metadynamics force and bias on grid
    kinetics=True,            # calculate importent metrics to get accurate kinetics
    verbose=True,             # print verbose output
)
the_bias.step_bias()

def print_output(the_md):
    print("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" % (
        the_md.step * the_md.dt * atomic_to_fs_times_sqrt_amu2au,
        the_md.coords[0],
        the_md.coords[1],
        the_md.epot,
        the_md.ekin,
        the_md.epot + the_md.ekin,
        the_md.temp,
    ))


print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % ("time [fs]", "x", "y", "E_pot", "E_kin", "E_tot", "Temp")
)
print_output(the_md)

while the_md.step < nsteps:
    the_md.step += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    if biased:
        the_md.forces += the_bias.step_bias()

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    if the_md.step % print_freq == 0:
        print_output(the_md)

    if the_md.step % traj_freq == 0:
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])

np.savez('coords.npz', x=x, y=y)
