import os
from sys import stdout
import numpy as np

from adaptive_sampling import units
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling import units

# ------------------------------------------------------------------------------------
# MD parameters
nsteps     = 10000001
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
mtd_hill_height   = 1.0     # MtD hill height in kJ/mol   
mtd_hill_std      = 0.1     # MtD hill width
mtd_wtm_temp      = 4200.0  # MtD Bias factor gamma
mtd_frequency     = 500     # MtD frequency of hill creation
output_freq       = 1000    # frequency of writing outputs

the_bias = WTM(
    the_md, 
    collective_var,         # collective variable
    well_tempered_temp=mtd_wtm_temp,
    hill_height=mtd_hill_height,
    hill_std=mtd_hill_std,
    hill_drop_freq=mtd_frequency,
    force_from_grid=True,     # accumulate metadynamics force and bias on grid
    equil_temp=temp,          # equilibrium temperature of simulation
    kinetics=True,            # calculate importent metrics to get accurate kinetics
    verbose=True,             # print verbose output
    output_freq=output_freq,  # frequency of updating outputs     
    f_conf=10000.0,           # confinement force of CV at boundaries
)
the_bias.step_bias()

def print_output(the_md):
    print("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" % (
        the_md.step * the_md.dt * units.atomic_to_fs_times_sqrt_amu2au,
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
N_mtdpots = 100
reweight_freq = int(nsteps/N_mtdpots)
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

    if the_md.step % reweight_freq == 0:
        biaspots.append(the_bias.metapot[0,:] * units.atomic_to_kJmol * units.kJ_to_kcal)

np.savez('coords.npz', x=x, y=y)
np.savez('results.npz', mtd_pots=biaspots)

