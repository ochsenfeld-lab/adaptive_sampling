# CONFIGURE PARAMETER
config_nsteps = 3e5
config_merge = 1.0          #Merging or not, np.inf is no merging
config_update_freq = 300
config_print_freq = 1000
config_approx_norm_factor = True
config_recursion_merge = False
config_verbose = False
config_md_print_freq = 10000

import os
from sys import stdout
from openmm import *
from openmm.app import *
from openmm.unit import *
from adaptive_sampling.sampling_tools import opes

import nglview as ngl
import mdtraj as pt
import numpy as np
import matplotlib.pyplot as plt

def run(nsteps: int=1000, T: float=300.0, dcd_freq: int=10, out_freq: int=10):
    
    # load system topology and coordinates from AMBER format
    prmtop = AmberPrmtopFile(f"../data/alanine-dipeptide.prmtop")
    crd = AmberInpcrdFile(f"../data/alanine-dipeptide.crd")

    # create the system and integrator 
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
    )
    platform = Platform.getPlatformByName('CPU')
    integrator = LangevinIntegrator(T * kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtosecond)

    # setup an simulation and run MD for nsteps
    simulation = Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(crd.positions)
    simulation.context.setVelocitiesToTemperature(T)
    simulation.reporters.append(DCDReporter('alanine-dipeptide-test.dcd', dcd_freq))
    simulation.reporters.append(StateDataReporter(
        stdout, 
        out_freq,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        separator='\t')
    )
    simulation.step(nsteps)

from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interface_openmm import AdaptiveSamplingOpenMM

# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms_psi    = [6, 8, 14, 16]  # backbone N-C-C-N torsion
cv_atoms_phi    = [4, 6, 8, 14]   # backbone C-N-C-C torsion
minimum         = -180.0          # minimum of the CV
maximum         = 180.0           # maximum of the CV
bin_width       = 5.0             # bin with along the CV

collective_var = [["torsion", cv_atoms_phi, minimum, maximum, bin_width]]
#collective_var_psi = [["torsion", cv_atoms_psi, minimum, maximum, bin_width]]

periodicity = [[-np.pi, np.pi]],#[-np.pi, np.pi]]

# ------------------------------------------------------------------------------------
# Setup OpenMM
prmtop = AmberPrmtopFile(f"../data/alanine-dipeptide.prmtop")
crd = AmberInpcrdFile(f"../data/alanine-dipeptide.crd")
system = prmtop.createSystem(
    nonbondedMethod=NoCutoff,
    constraints=HBonds,
)

# remove center of mass motion
#cmm_force = CMMotionRemover()
#cmm_force.setFrequency(0)
#system.addForce(cmm_force)

# Initialize the `AdaptiveSamplingOpenMM` interface to couple the OpenMM simulaiton to an bias potential
# the Openmm `simulation` object is set up internally, but can still be modified by calling `the_md.simulation` or `the_md.integrator`
the_md = AdaptiveSamplingOpenMM(
    crd.positions,
    prmtop.topology,
    system,
    dt=2.0,                                       # timestep in fs
    equil_temp=300.0,                             # temperature of simulation
    langevin_damping=1.0,                         # langevin damping in 1/ps
    cv_atoms=np.unique(cv_atoms_phi+cv_atoms_psi) # specifying CV atoms significantly speeds up simulation of large systems
)                                                 # as the bias force will only be set for those
the_md.integrator.setConstraintTolerance(0.00001)

# Append OpenMM reporters to simulation for output 
the_md.simulation.reporters.append(DCDReporter('alanine-dipeptide.dcd', 1000))
the_md.simulation.reporters.append(StateDataReporter(
    stdout, 
    config_md_print_freq, # Print MD Output
    step=True,    
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    temperature=True,
    speed=False,
    separator='\t')
)

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
output_freq       = 1000    # frequency of writing outputs
kernel_std        = np.array([5.0])

the_bias = OPES(
    kernel_std,
    the_md,
    collective_var,
    output_freq=output_freq,
    equil_temp=300.0,
    energy_barr = 20.0,
    merge_threshold=config_merge,
    approximate_norm=config_approx_norm_factor,
    verbose=config_verbose,
    recursion_merge=config_recursion_merge,
    update_freq = config_update_freq,
)
the_md.set_sampling_algorithm(the_bias) # to take affect the sampling algorithm has to be set in the MD interface

# Warning: this may take a while!
if True:
    os.system("rm CV_traj.dat restart_opes.npz")
    the_md.run(nsteps=config_nsteps) # 500000 * 2 fs = 1 ns