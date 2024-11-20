import numpy as np
import os, sys, time
from openmm import *
from openmm.app import *
from openmm.unit import *
from adaptive_sampling.sampling_tools.opes import OPES
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interface_openmm import AdaptiveSamplingOpenMM

# ------------------------------------------------------------------------------------
# MD Parameters
nsteps = 1e6  # number of steps
energy_barrier = 20  # energy barrier in kJ/mol
traj_freq = 10  # frequency of writing trajectory
print_freq = 5000  # frequency of printing output
biased = True  # enables biased simulation

# ------------------------------------------------------------------------------------
# Setup CV
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
temp = 300.0

# Initialize the `AdaptiveSamplingOpenMM` interface to couple the OpenMM simulaiton to an bias potential
# the Openmm `simulation` object is set up internally, but can still be modified by calling `the_md.simulation` or `the_md.integrator`
the_md = AdaptiveSamplingOpenMM(
    crd.positions,
    prmtop.topology,
    system,
    dt=2.0,  # timestep in fs
    equil_temp=temp,  # temperature of simulation
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
        sys.stdout,
        print_freq,  # Print MD Output
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
opes_hill_std = [8.6, 8.6]  # OPES hill width
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
    kernel_std=np.asarray(opes_hill_std),
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
the_md.set_sampling_algorithm(
    the_bias
)  # to take affect the sampling algorithm has to be set in the MD interface

# --------------------------------------------------------------------------------------
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


# Warning: this may take a while!
if biased:
    if os.path.isfile("CV_traj.dat"):
        print("Removing old trajectory")
        os.system("rm CV_traj.dat")
    if os.path.isfile("restart_opes.npz"):
        os.system("rm restart_opes.npz")
    the_md.run(nsteps=nsteps)
