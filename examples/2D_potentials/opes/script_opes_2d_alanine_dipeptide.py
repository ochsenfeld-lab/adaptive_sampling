import os

from sys import stdout
import numpy as np
import mdtraj as md

from openmm import *
from openmm.app import *
from openmm.unit import *

from adaptive_sampling import units
from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interface_openmm import AdaptiveSamplingOpenMM

# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms_psi    = [6, 8, 14, 16]  # backbone N-C-C-N torsion
cv_atoms_phi    = [4, 6, 8, 14]   # backbone C-N-C-C torsion
minimum         = -180.0          # minimum of the CV
maximum         = 180.0           # maximum of the CV
bin_width       = 5.0             # bin with along the CV

collective_var = [
    ["torsion", cv_atoms_phi, minimum, maximum, bin_width],
    ["torsion", cv_atoms_psi, minimum, maximum, bin_width],
]

periodicity = [[-np.pi, np.pi], [-np.pi, np.pi]]

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
the_md.simulation.reporters.append(DCDReporter(f'alanine-dipeptide.dcd', 10))
the_md.simulation.reporters.append(StateDataReporter(
    stdout, 
    1000,
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
opes_hill_std  = [np.degrees(0.15), np.degrees(0.15)] # initial standard deviation of hills
opes_gamma     = 15.0               # Well-tempered bias factor
opes_frequency = 500                # frequency of hill creation
opes_barrier   = 50.0               # maximum barrier in kJ/mol

the_bias = OPES(
    the_md,
    collective_var,
    kernel_std=np.asarray(opes_hill_std),
    update_freq=opes_frequency,
    bias_factor=opes_gamma,
    energy_barr=opes_barrier,
    bandwidth_rescaling=True,       # shrink kernels over time
    adaptive_std=False,             # adaptive sigma estimation
    adaptive_std_freq=10,           # adaptive sigma estimation
    explore=False,                  # OPES explore
    merge_threshold=1.0,            # kernel merge threshold
    recursive_merge=True,           # merge kernels recursively
    normalize=True,                 # normalize prob dist over explored space
    approximate_norm=True,          # linear scaling norm factor approximation
    force_from_grid=True,           # read OPES force from numerical grid 
    output_freq=1000,             
    equil_temp=300.0,
    verbose=True,
    f_conf=0.0,
    periodicity=periodicity,
    kinetics=True,
)
the_md.set_sampling_algorithm(the_bias) # to take affect the sampling algorithm has to be set in the MD interface

nsteps = 5000000
N_reweights = 100
steps_per_iter = int(nsteps/N_reweights)
biaspots = []

for _ in range(N_reweights):
    the_md.run(nsteps=steps_per_iter, traj_file=f"CV_traj.dat")
    biaspots.append(the_bias.pmf * units.atomic_to_kJmol * units.kJ_to_kcal)
    np.save(f"biaspots.npy", biaspots)

the_md.run(nsteps=1, traj_file=f"CV_traj.dat")
