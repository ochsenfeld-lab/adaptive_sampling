# -------------------------------------------------------------------------------------
### Script for OPES test runs in predefined toy potentials
# -------------------------------------------------------------------------------------
import numpy as np
import os, sys, time
from adaptive_sampling.units import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.sampling_tools.opes import OPES

# ------------------------------------------------------------------------------------
# MD Parameters

nsteps = 1e6  # number of steps
energy_barrier = 20  # energy barrier in kJ/mol
traj_freq = 10  # frequency of writing trajectory
print_freq = 1000  # frequency of printing output
biased = True  # enables biased simulation

# ------------------------------------------------------------------------------------
# Setup CV

# remove old output
if os.path.isfile("CV_traj.dat"):
    print("Removing old trajectory")
    os.system("rm CV_traj.dat")
if os.path.isfile("restart_opes.npz"):
    os.system("rm restart_opes.npz")

cv_atoms = []  # not needed for 2D potentials
min_1 = -0.3  # minimum of CV 1
max_1 = 2.5  # maximum of CV 1
bin_width_1 = 0.05  # bin width along CV 1
min_2 = -1.5  # minimum of CV 2
max_2 = 1.5  # minimum of CV 2
bin_width_2 = 0.05  # bin width along CV 2

collective_var = [
    ["x", cv_atoms, min_1, max_1, bin_width_1],
    #    ["y", cv_atoms, min_2, max_2, bin_width_2],
]

periodicity = [None]

grid_1 = np.arange(min_1, max_1, bin_width_1)
grid_2 = np.arange(min_2, max_2, bin_width_2)

# ------------------------------------------------------------------------------------
# Setup MD
mass = 10.0  # mass of particle in a.u.
seed = np.random.randint(1000)  # random seed
dt = 1.0e0  # stepsize in fs
temp = 300.0  # temperature in K

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
opes_hill_std = 0.07  # OPES hill width
opes_bias_factor = None  # OPES Bias factor gamma
opes_frequency = 500  # OPES frequency of hill creation
opes_adaptive_std = False  # Adaptive estimate of kernel standard deviation
opes_adaptive_std_stride = (
    10  # time for estimate of kernel std on units of `opes_frequency`
)
opes_output_freq = 1000  # frequency of writing outputs
opes_explore = False  # enable explore mode

the_bias = OPES(
    the_md,
    collective_var,
    kernel_std=np.asarray([opes_hill_std]),
    energy_barr=energy_barrier,
    bias_factor=opes_bias_factor,
    bandwidth_rescaling=True,
    adaptive_std=opes_adaptive_std,
    adaptive_std_stride=opes_adaptive_std_stride,
    explore=opes_explore,
    update_freq=opes_frequency,
    periodicity=periodicity,
    normalize=True,
    approximate_norm=True,
    merge_threshold=1.0,
    recursive_merge=True,
    force_from_grid=False,
    output_freq=opes_output_freq,
    f_conf=0.0,  # confinement force of CV at boundaries
    equil_temp=temp,  # equilibrium temperature of simulation
    kinetics=True,  # calculate importent metrics to get accurate kinetics
    verbose=True,  # print verbose output
)
the_bias.step_bias()

# --------------------------------------------------------------------------------------
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
            t,
            the_bias.kernel_std[-1][0] if len(the_bias.kernel_std) > 0 else 0.0,
        )
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# Run MD
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
        "n kernel",
        "wall time",
        "last std",
    )
)
print_output(the_md, the_bias, 0)
x, y, kernel_number, bias_potentials = [], [], [], []
while the_md.step < nsteps:
    the_md.step += 1

    the_md.propagate(langevin=True)
    the_md.calc()

    t0 = time.perf_counter()
    if biased:
        the_md.forces += the_bias.step_bias()
        bias_potentials.append(the_bias.bias_potential)
        if the_md.step % the_bias.update_freq == 0:
            kernel_number.append(len(the_bias.kernel_center))

    t = time.perf_counter() - t0
    the_md.up_momenta(langevin=True)
    the_md.calc_etvp()

    if the_md.step % opes_output_freq == 0:
        print_output(the_md, the_bias, t)

    if the_md.step % traj_freq == 0:
        x.append(the_md.coords[0])
        y.append(the_md.coords[1])

# Save full trajectory for alternative reweighting
np.savez("full_traj.npz", x=x, y=y)
np.savez("results.npz", opes_pots=bias_potentials)
