# ------------------------------------------------------------------------------------
# Script for OPESeABF 1d double well simulations
# ------------------------------------------------------------------------------------
import numpy as np

config_nsteps = 5e6  # Number of simulation steps
config_explore = True  # Enable Exploration mode
config_adaptive_sigma = False  # Enable adaptive sigma calculation according to welford
config_unbiased_time = 100  # Determine how many update freq steps an unbiased simulation is run to approximate sigma 0
config_input = True  # If False enable unbiased simulation to estimate sigma 0
config_fixed_sigma = (
    False  # Disable bandwidth rescaling and use input std for all kernels
)
config_merge = 1.0  # Merging or not, np.inf is no merging
config_update_freq = 500  # Frequency in which kernels are placed
config_recursion_merge = True  # Enable recursive merging
config_approx_norm_factor = True  # Enable approximation of norm factor
config_exact_norm_factor = (
    True  # Enable exact norm factor, if both activated, exact is used every 100 updates
)
config_f_conf = 1000.0  # Confinment force to keep system in boundaries
config_bias_factor = None  # Direct setting of bias factor, if calculation from energy is wanted put in 'None'
config_energy_barr = 50.0  # Energy barrier to overcome
config_verbose = False  # Enable for debugging
config_enable_eabf = False  # Enable eABF
config_enable_opes = True  # Enable eOPES
config_traj_reduction = 10  # Reduce length of trajectories to compute MBAR
config_md_print_freq = 5000  # Print frequency in .grid file
config_nsteps_output = 5000  # Frequency in which outfile for postprocessing is written
# ------------------------------------------------------------------------------------
import os
from sys import stdout
import sys
import time
import nglview as ngl
import mdtraj as pt
import matplotlib.pyplot as plt
from adaptive_sampling.sampling_tools.opeseabf import OPESeABF
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interfaceMD_2D import MD
from adaptive_sampling.units import *
from openmm import *
from openmm.app import *
from openmm.unit import *


def run(nsteps: int = 1000, T: float = 300.0, dcd_freq: int = 10, out_freq: int = 10):

    # load system topology and coordinates from AMBER format
    prmtop = AmberPrmtopFile(f"../data/alanine-dipeptide.prmtop")
    crd = AmberInpcrdFile(f"../data/alanine-dipeptide.crd")

    # create the system and integrator
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
    )
    platform = Platform.getPlatformByName("CPU")
    integrator = LangevinIntegrator(
        T * kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtosecond
    )

    # setup an simulation and run MD for nsteps
    simulation = Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(crd.positions)
    simulation.context.setVelocitiesToTemperature(T)
    simulation.reporters.append(DCDReporter("alanine-dipeptide-test.dcd", dcd_freq))
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            out_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            separator="\t",
        )
    )
    simulation.step(nsteps)


from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interface_openmm import AdaptiveSamplingOpenMM

# --------------------------------------------------------------------------------------
# Define collective variables
cv_atoms_psi = [6, 8, 14, 16]  # backbone N-C-C-N torsion
cv_atoms_phi = [4, 6, 8, 14]  # backbone C-N-C-C torsion
minimum = -180.0  # minimum of the CV
maximum = 180.0  # maximum of the CV
bin_width = 5.0  # bin with along the CV

collective_var = [
    ["torsion", cv_atoms_phi, minimum, maximum, bin_width],
    ["torsion", cv_atoms_psi, minimum, maximum, bin_width],
]

periodicity = [[-np.pi, np.pi], [-np.pi, np.pi]]

# ------------------------------------------------------------------------------------
# Setup OpenMM
prmtop = AmberPrmtopFile(
    f"/home/rschiller/Code/adaptive_sampling/tutorials/data/alanine-dipeptide.prmtop"
)
crd = AmberInpcrdFile(
    f"/home/rschiller/Code/adaptive_sampling/tutorials/data/alanine-dipeptide.crd"
)
system = prmtop.createSystem(
    nonbondedMethod=NoCutoff,
    constraints=HBonds,
)

# Initialize the `AdaptiveSamplingOpenMM` interface to couple the OpenMM simulaiton to an bias potential
# the Openmm `simulation` object is set up internally, but can still be modified by calling `the_md.simulation` or `the_md.integrator`
the_md = AdaptiveSamplingOpenMM(
    crd.positions,
    prmtop.topology,
    system,
    dt=2.0,  # timestep in fs
    equil_temp=300.0,  # temperature of simulation
    langevin_damping=1.0,  # langevin damping in 1/ps
    cv_atoms=np.unique(
        cv_atoms_phi + cv_atoms_psi
    ),  # specifying CV atoms significantly speeds up simulation of large systems
)  # as the bias force will only be set for those
the_md.integrator.setConstraintTolerance(0.00001)

# Append OpenMM reporters to simulation for output
the_md.simulation.reporters.append(DCDReporter("alanine-dipeptide.dcd", 1000))
the_md.simulation.reporters.append(
    StateDataReporter(
        stdout,
        config_md_print_freq,  # Print MD Output
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        separator="\t",
    )
)

# ------------------------------------------------------------------------------------
# Setup the sampling algorithm
eabf_ext_sigma = 2.0  # thermal width of coupling between CV and extended variable
eabf_ext_mass = 20.0  # mass of extended variable in a.u.
abf_nfull = 500  # number of samples per bin when abf force is fully applied
output_freq = 1000  # frequency of writing outputs
kernel_std = (
    np.array([8.6, 8.6]) if config_input else np.array([None, None])
)  # std in angstrom or degree

the_bias = OPESeABF(
    eabf_ext_sigma,
    eabf_ext_mass,
    the_md,
    collective_var,  # collective variable
    enable_eabf=config_enable_eabf,
    enable_opes=config_enable_opes,
    output_freq=1000,  # frequency of writing outputs
    nfull=abf_nfull,
    equil_temp=300.0,  # equilibrium temperature of simulation
    force_from_grid=True,  # accumulate metadynamics force and bias on grid
    kinetics=True,  # calculate importent metrics to get accurate kinetics
    kernel_std=kernel_std,
    adaptive_sigma=config_adaptive_sigma,
    unbiased_time=config_unbiased_time,
    fixed_sigma=config_fixed_sigma,
    explore=config_explore,
    periodicity=periodicity,
    energy_barr=config_energy_barr,
    merge_threshold=config_merge,
    approximate_norm=config_approx_norm_factor,
    exact_norm=config_exact_norm_factor,
    verbose=config_verbose,
    recursion_merge=config_recursion_merge,
    update_freq=config_update_freq,
    f_conf=config_f_conf,
    bias_factor=config_bias_factor,
)

the_md.set_sampling_algorithm(
    the_bias
)  # to take affect the sampling algorithm has to be set in the MD interface

# ------------------------------------------------------------------------------------
# Warning: this may take a while!
if True:
    os.system("rm CV_traj.dat restart_opes.npz")
    the_md.run(
        nsteps=config_nsteps,
        stabilize=True,
        stabilizer_threshold=[np.pi, np.pi],
        write_output=False,
    )  # 500000 * 2 fs = 1 ns
    # nsteps_output = config_nsteps_output
    # with open('kernels.dat', 'w') as out:
    #    out.write('# kernels\tN_eff\tZ_n\t# merged')

    # for i in range(nsteps_output):
    #    the_md.run(nsteps=int(config_nsteps/nsteps_output)) # 500000 * 2 fs = 1 ns
    #    with open('kernels.dat', 'a') as out:
    #        out.write(f'{len(the_bias.kernel_center)}\t{the_bias.n_eff}\t{the_bias.norm_factor}\t{the_bias.merge_count}')


print("========================================================================")
print("### Simulation finished, start post processing... ###")
print("========================================================================")

from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.processing_tools.utils import autocorr
from adaptive_sampling.processing_tools.utils import ipce


cv_traj = np.loadtxt("CV_traj.dat", skiprows=1)
cv_x = np.array(cv_traj[:, 1])
cv_y = np.array(cv_traj[:, 2])
cv_la_x = np.array(cv_traj[:, 3])
cv_la_y = np.array(cv_traj[:, 4])

corr_x = autocorr(cv_x)
corr_y = autocorr(cv_y)

tau_x = ipce(corr_x)
tau_y = ipce(corr_y)

print("Autocorrelation times are ", tau_x, " for x and ", tau_y, " for y")

sys.stdout.flush()

traj_reduction = config_traj_reduction
cv_x = cv_x[::traj_reduction]
cv_y = cv_y[::traj_reduction]
cv_la_x = cv_la_x[::traj_reduction]
cv_la_y = cv_la_y[::traj_reduction]

ext_sigma = 2.0

# grid for free energy profile can be different than during sampling
minimum_x = -180.0
maximum_x = 180.0
minimum_y = -180.0
maximum_y = 180.0
bin_width = 2.0

# get tuple matrix of centers
mbar_grid_x = np.arange(minimum_x, maximum_x, bin_width)
mbar_grid_y = np.arange(minimum_y, maximum_y, bin_width)
mbar_xx, mbar_yy = np.meshgrid(mbar_grid_x, mbar_grid_y)
# mbar_xy = np.array(list(zip(mbar_xx.ravel(),mbar_yy.ravel())), dtype=('i4,i4')).reshape(mbar_xx.shape)
mbar_xy = np.array(list(zip(mbar_xx.ravel(), mbar_yy.ravel())))

# zip cv`s of both dimensions as well as extended coordinates
mbar_cv = np.array(list(zip(cv_x, cv_y)))
mbar_la = np.array(list(zip(cv_la_x, cv_la_y)))
# run MBAR and compute free energy profile and probability density from statistical weights
print("========================================================================")
print("Initialize MBAR...")
print("Getting windows...")

sys.stdout.flush()

traj_list, indices, meta_f = mbar.get_windows(
    mbar_xy, mbar_cv, mbar_la, ext_sigma, equil_temp=300.0, dx=np.array([2.0, 2.0])
)
print("========================================================================")
print("Build Boltzmann...")
sys.stdout.flush()

exp_U, frames_per_traj = mbar.build_boltzmann(
    traj_list, meta_f, equil_temp=300.0, periodicity=[[-180.0, 180.0], [-180.0, 180.0]]
)

print("Max Boltzmann is ", exp_U.max())
print("Min Boltzmann is ", exp_U.min())
sys.stdout.flush()

print("========================================================================")
print("Initialize MBAR")
print("")
print("")
sys.stdout.flush()

weights = mbar.run_mbar(
    exp_U,
    frames_per_traj,
    max_iter=20000,
    conv=1.0e-4,
    conv_errvec=1.0,
    outfreq=100,
    device="cpu",
)

print("MBAR finished")
print("========================================================================")
print("")
print("")

print("Calculate PMF from MBAR")
sys.stdout.flush()

mbar_cv = mbar_cv[indices]
RT = R_in_SI * 300.0 / 1000.0

dx = np.array([2.0, 2.0])
dx2 = dx / 2
rho = np.zeros(shape=(len(mbar_xy),), dtype=float)
for ii, center in enumerate(mbar_xy):
    indices = np.where(
        np.logical_and(
            (mbar_cv >= center - dx2).all(axis=-1),
            (mbar_cv < center + dx2).all(axis=-1),
        )
    )
    rho[ii] = weights[indices].sum()

rho /= rho.sum() * np.prod(dx)
pmf = -RT * np.log(rho, out=np.full_like(rho, np.NaN), where=(rho != 0))
pmf = np.ma.masked_array(pmf, mask=np.isnan(pmf))
pmf_mbar = pmf.reshape(mbar_xx.shape)
rho_mbar = rho

print("PMF aquired, done!")

print("")
print("")
print("Postprocessing finished successfully.")

np.savez(
    "mbar_data.npz",
    weights=weights,
    indices=indices,
    pmf_mbar=pmf_mbar,
    rho_mbar=rho_mbar,
)

print("Saved!")
