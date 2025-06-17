import os
from sys import stdout
import numpy as np

from adaptive_sampling import units
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *

# ------------------------------------------------------------------------------------
# MD parameters
nsteps      = 505000
N_reweights = 100
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
minimum         = -0.2            # minimum of the CV
maximum         = 2.5             # maximum of the CV
bin_width       = 0.05            # bin with along the CV

collective_var = [["x", cv_atoms, minimum, maximum, bin_width]]

# ------------------------------------------------------------------------------------
# setup MD
mass      = 10.0   # mass of particle in a.u.
seed      = np.random.randint(1000) # random seed
dt        = 1.0e0  # stepsize in fs
temp      = 300.0  # temperature in K

coords_in = [np.random.normal(2.1, 0.07), np.random.normal(0, 0.15)]
print(f"STARTING MD FROM {coords_in}")

the_md = MD(
    mass_in=mass,
    coords_in=coords_in,
    potential="4",
    dt_in=dt,
    target_temp_in=temp,
    seed_in=seed,
)
the_md.calc_init()
the_md.calc_etvp()

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
eabf_ext_sigma           = None    # thermal width of coupling between CV and extended variable 
eabf_ext_mass            = 20.0    # mass of extended variable in a.u.
adaptive_coupling_stride = 5000    # number of steps of initial unbiased MD to estimate ext_sigma
adaptive_coupling_scaling = 0.5    # scaling factr of estimation of ext_sigma
abf_nfull                 = 500    # number of samples per bin when abf force is fully applied

opes_hill_std            = None    # OPES hill width, if None, estimated from opes_adaptive_std_stride*opes_frequency steps
opes_bias_factor         = 15.0    # OPES Bias factor gamma
opes_frequency           = 500     # OPES frequency of hill creation
opes_adaptive_std        = False   # Adaptive estimate of kernel standard deviation
opes_adaptive_std_stride = 10      # time for estimate of kernel std on units of `opes_frequency`
opes_output_freq         = 1000    # frequency of writing outputs

the_bias = OPESeABF(
    the_md, 
    collective_var,
    ext_sigma=eabf_ext_sigma,
    ext_mass=eabf_ext_mass,
    adaptive_coupling_stride=adaptive_coupling_stride,
    adaptive_coupling_scaling=adaptive_coupling_scaling,
    nfull=abf_nfull,
    enable_abf=True,
    kernel_std=np.asarray([opes_hill_std]),
    energy_barr=127.218,
    bias_factor=opes_bias_factor,
    bandwidth_rescaling=True,
    adaptive_std=opes_adaptive_std,
    adaptive_std_stride=opes_adaptive_std_stride,
    explore=False,
    update_freq=opes_frequency,
    normalize=True,
    approximate_norm=True,
    merge_threshold=1.0,
    recursive_merge=True,
    force_from_grid=True,
    output_freq=opes_output_freq,       
    f_conf=0.0,             # confinement force of CV at boundaries
    equil_temp=temp,        # equilibrium temperature of simulation
    kinetics=True,          # calculate importent metrics to get accurate kinetics
    verbose=True,           # print verbose output
)
the_bias.step_bias()

# --------------------------------------------------------------------------------------
def print_output(the_md):
    print("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" % (
        the_md.step * the_md.dt * atomic_to_fs,
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
biaspots = []
reweight_freq = int(nsteps/N_reweights)
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
        biaspots.append(the_bias.bias_potential * atomic_to_kJmol * kJ_to_kcal)

np.savez('coords.npz', x=x, y=y)
np.savez('results.npz', opes_pots=biaspots)

