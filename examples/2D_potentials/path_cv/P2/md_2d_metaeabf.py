#!/usr/bin/env python
import time
from adaptive_sampling.sampling_tools import WTM, eABF, Reference, WTMeABF
from adaptive_sampling.interface.interfaceMD_2D import *
from adaptive_sampling.units import *

################# Imput Section ####################
# MD
seed = 42
nsteps = 100000  # number of MD steps
dt = 5.0e0  # stepsize in fs
target_temp = 300.0  # Kelvin
mass = 10.0  # a.u.
potential = "2"

# eABF
dev = [
    {
        "guess_path": "guess_path.xyz",
        "metric": "2d",
        "verbose": True,
        "n_interpolate": 0,
        "adaptive": True,
        "update_interval": 1000,
        "half_life": 100,
        }
]
tube = False

ats = [["path", dev, 0.0, 1.0, 0.05]]
conf = [["GPath_tube", dev, 0.0, 0.0, 0.0]]
N_full = 100

step_count = 0
coords = [-50.0, -30.0]
the_md = MD(
    mass_in=mass,
    coords_in=coords,
    potential=potential,
    dt_in=dt,
    target_temp_in=target_temp,
    seed_in=seed,
)
the_abm = WTMeABF(
    0.05,
    40.0,
    0.5,
    0.1,
    the_md,
    ats,
    nfull=100,
    output_freq=10,
    f_conf=1000.0,
    equil_temp=300.0,
    multiple_walker=False,
)
if tube:
    the_conf = Reference(
        the_md,
        conf,
        nfull=100,
        output_freq=1000,
        f_conf=50.0,
        equil_temp=300.0,
        multiple_walker=False,
    )

# the_abm.restart()
the_md.calc_init()
the_abm.step_bias()
if tube:
    the_conf.step_bias()
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
    t = time.perf_counter()
    the_md.calc()
    calc_time = time.perf_counter() - t
    t = time.perf_counter()
    the_md.forces += the_abm.step_bias()
    bias_time = time.perf_counter() - t 
    if tube:
        the_md.forces += the_conf.step_bias()

    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    print(
        "%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"
        % (
            the_md.step * the_md.dt * atomic_to_fs,
            the_md.coords[0],
            the_md.coords[1],
            the_md.epot,
            the_md.ekin,
            the_md.epot + the_md.ekin,
            the_md.temp,
            calc_time,
            bias_time,
        )
    )
    if not the_md.step % 1000 and dev[0]["adaptive"]:
        the_abm.the_cv.pathcv.write_path(filename=f'path_{step_count}.npy')
