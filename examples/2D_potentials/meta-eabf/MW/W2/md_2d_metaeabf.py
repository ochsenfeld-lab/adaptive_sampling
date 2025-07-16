#!/usr/bin/env python
import random
from adaptive_sampling.sampling_tools.metaeabf import WTMeABF
from adaptive_sampling.interface.interfaceMD_2D import *
from adaptive_sampling.units import *

################# Imput Section ####################
# MD
seed = random.randint(0, 10000)
nsteps = 100000  # number of MD steps
dt = 5.0e0  # stepsize in fs
target_temp = 300.0  # Kelvin
mass = 10.0  # a.u.
potential = "1"

# eABF
ats = [["x", [], 70.0, 170.0, 2.0]]
N_full = 100

step_count = 0
coords = [71.0, 0]
the_md = MD(
    mass_in=mass,
    coords_in=coords,
    potential=potential,
    dt_in=dt,
    target_temp_in=target_temp,
    seed_in=seed,
)
the_abm = WTMeABF(
    2.0,
    20.0,
    2.0,
    4.0,
    the_md,
    ats,
    hill_drop_freq=20,
    output_freq=1000,
    f_conf=1000.0,
    equil_temp=300.0,
    force_from_grid=True,
    multiple_walker=True,
)
# the_abm.restart()

step_dict = {
    "mw_file": "../shared_bias1",
    "sync_interval": 5000,
    "output_file": "wtmeabf1.out",
    "traj_file": "CV_traj1.dat",
    "restart_file": "restart_wtmeabf1",
}

the_md.calc_init()
the_md.forces += the_abm.step_bias(**step_dict)
the_md.calc_etvp()

################# the MD loop ####################
print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % ("time [fs]", "x", "y", "E_pot", "E_kin", "E_tot", "T")
)
print(
    "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"
    % (
        the_md.step * the_md.dt * atomic_to_fs_times_sqrt_amu2au,
        the_md.coords[0],
        the_md.coords[1],
        the_md.epot,
        the_md.ekin,
        the_md.epot + the_md.ekin,
        the_md.temp,
    )
)

while step_count < nsteps:
    the_md.step += 1
    step_count += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    the_md.forces += the_abm.step_bias(**step_dict)

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs_times_sqrt_amu2au,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_md.epot + the_md.ekin,
            the_md.temp,
        )
    )
