# ------------------------------------------------------------------------------------
# CONFIGURE PARAMETER
# ------------------------------------------------------------------------------------
import numpy as np

config_nsteps = 5e6  # Number of simulation steps
config_explore = True  # Enable Exploration mode
config_adaptive_sigma = True  # Enable adaptive sigma calculation according to welford
config_unbiased_time = 100  # Determine how many update freq steps an unbiased simulation is run to approximate sigma 0
config_input = False  # If False enable unbiased simulation to estimate sigma 0
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
config_md_print_freq = 5000  # Print frequency in .grid file
config_nsteps_output = 5000  # Frequency in which outfile for postprocessing is written
# ------------------------------------------------------------------------------------
# Load packages and set up simulation
import os
from sys import stdout
from openmm import *
from openmm.app import *
from openmm.unit import *
from adaptive_sampling.sampling_tools import opes

import nglview as ngl
import mdtraj as pt
import matplotlib.pyplot as plt


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

# ------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
output_freq = 1000  # frequency of writing outputs
kernel_std = (
    np.array([8.6, 8.6]) if config_input else np.array([None, None])
)  # std in angstrom or degree

the_bias = OPES(
    the_md,
    collective_var,
    kernel_std=kernel_std,
    adaptive_sigma=config_adaptive_sigma,
    unbiased_time=config_unbiased_time,
    fixed_sigma=config_fixed_sigma,
    explore=config_explore,
    periodicity=periodicity,
    output_freq=output_freq,
    equil_temp=300.0,
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

# Warning: this may take a while!
if True:
    os.system("rm CV_traj.dat restart_opes.npz")
    nsteps_output = config_nsteps_output
    with open("kernels.dat", "w") as out:
        out.write("# kernels\tN_eff\tZ_n\t# merged")
    for i in range(nsteps_output):
        the_md.run(nsteps=int(config_nsteps / nsteps_output))  # 500000 * 2 fs = 1 ns
        with open("kernels.dat", "a") as out:
            out.write(
                f"{len(the_bias.kernel_center)}\t{the_bias.n_eff}\t{the_bias.norm_factor}\t{the_bias.merge_count}"
            )


# weighted PMF history
if True:
    grid_x = np.arange(-180.0, 180.0)
    grid_y = np.arange(-180.0, 180.0)
    cv_traj = np.loadtxt("CV_traj.dat", skiprows=1)
    cv_phi = np.array(cv_traj[:, 1])
    cv_psi = np.array(cv_traj[:, 2])
    cv_pot = np.array(cv_traj[:, 5])
    pmf_weight_history, scattered_time = the_bias.weighted_pmf_history2d(
        cv_phi, cv_psi, cv_pot, grid_x, grid_y, hist_res=50
    )
    np.savez("pmf_hist.npz", pmf_weight_history, scattered_time)
