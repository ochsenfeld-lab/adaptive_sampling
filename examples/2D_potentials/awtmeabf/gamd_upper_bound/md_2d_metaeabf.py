#!/usr/bin/env python
from adaptive_sampling.sampling_tools.awtmeabf import aWTMeABF
from adaptive_sampling.interface.interfaceMD_2D import *

bohr2angs = 0.52917721092e0

################# Imput Section ####################

# MD
seed = 42
nsteps = 210000  # number of MD steps
dt = 5.0e0  # stepsize in fs
target_temp = 300.0  # Kelvin
mass = 10.0  # a.u.
potential = "1"

# eABF
ats = [["x", [], 70.0, 170.0, 2.0]]
N_full = 100

step_count = 0
coords = [80.0, 0]
the_md = MD(
    mass_in=mass,
    coords_in=coords,
    potential=potential,
    dt_in=dt,
    target_temp_in=target_temp,
    seed_in=seed,
)
the_abm = aWTMeABF(
    0.0005,
    1000,
    10000,
    the_md,
    ats,
    ext_sigma=2.0,
    ext_mass=20.0,
    hill_height=2.0,
    hill_std=4.0,
    hill_drop_freq=100,
    do_wtm=True,
    amd_method="gamd_lower",
    output_freq=1000,
    f_conf=100,
    equil_temp=300.0,
)
#the_abm.restart('restart_awtmeabf')

the_md.calc_init()
the_abm.step_bias()
the_md.calc_etvp()


################# the MD loop ####################
print(
    "%11s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s"
    % ("time [fs]", "x", "y", "E_pot", "E_kin", "E_tot", "T")
)
print(
    "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"
    % (
        the_md.step * the_md.dt * atomic_to_fs,
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

    the_md.forces += the_abm.step_bias()

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_md.epot + the_md.ekin,
            the_md.temp,
        )
    )
