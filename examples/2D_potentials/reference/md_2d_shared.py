#!/usr/bin/env python
import time
from InterfaceMD_2D import *
from adaptive_sampling.sampling_tools.reference import Reference

bohr2angs = 0.52917721092e0

################# Imput Section ####################

# MD
seed = 42
nsteps = 1000  # number of MD steps
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
the_abm = Reference(the_md, ats, 300.0, output_freq=1000, f_conf=100)
# the_abm.restart()

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
        the_md.step * the_md.dt * it2fs,
        the_md.coords[0],
        the_md.coords[1],
        the_md.epot,
        the_md.ekin,
        the_md.epot + the_md.ekin,
        the_md.temp,
    )
)

while step_count < nsteps:
    start_loop = time.perf_counter()
    the_abm.shared_bias()
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
            the_md.step * the_md.dt * it2fs,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_md.epot + the_md.ekin,
            the_md.temp,
        )
    )
