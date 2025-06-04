#!/usr/bin/env python
from adaptive_sampling.sampling_tools.metaeabf import WTMeABF
from adaptive_sampling.interface.interfaceMD_2D import *
from adaptive_sampling.units import *

################# Imput Section ####################
# MD
seed = 1895912917
nsteps = 40000000  # number of MD steps
dt = 5.0e0  # fs
target_temp = 300.0  # K
mass = 10.0
potential = "2"

# eABF
ats = [["x", [], -50.0, 50.0, 2.0]]
N_full = 100
f_conf = 500.0

# WTM-eABF
ext_sigma = 2.0
ext_mass = 20.0
mtd_std = 6.0
grid = True
height = 1.0
hill_drop_freq = 20
WT_dT = 4000

#################### Pre-Loop ####################
step_count = 0
coords = [0.0, 0.0]
the_md = MD(
    mass_in=mass,
    coords_in=coords,
    potential=potential,
    dt_in=dt,
    target_temp_in=target_temp,
    seed_in=seed,
)
the_abm = WTMeABF(
    the_md,
    ats,
    nfull=N_full,
    ext_sigma=ext_sigma,
    ext_mass=ext_mass,
    hill_height=height,
    hill_std=mtd_std,
    hill_drop_freq=hill_drop_freq,
    well_tempered_temp=WT_dT,
    force_from_grid=True,
    f_conf=f_conf,
    output_freq=1000,
    seed_in=seed,
)

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

#################### the MD loop ####################
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
