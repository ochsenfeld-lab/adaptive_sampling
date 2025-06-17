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

collective_var = [["torsion", cv_atoms_phi, minimum, maximum, bin_width]]

periodicity = [[-np.pi, np.pi]]

# ------------------------------------------------------------------------------------
# Setup OpenMM
prmtop = AmberPrmtopFile(f"../alanine-dipeptide.prmtop")
crd = AmberInpcrdFile(f"../alanine_c7ax.crd")
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
mtd_hill_height   = 1.2     # MtD hill height    
mtd_hill_std      = np.degrees(0.35) # MtD hill width
mtd_well_tempered = 4200.0  # Well-tempered temperature
mtd_frequency     = 500     # frequency of hill creation

the_bias = WTM(
    the_md, 
    collective_var,         # collective variable
    well_tempered_temp=mtd_well_tempered,
    hill_drop_freq=mtd_frequency,
    hill_height=mtd_hill_height,
    hill_std=mtd_hill_std,
    force_from_grid=True,   # accumulate metadynamics force and bias on grid
    output_freq=1000,       # frequency of writing outputs
    f_conf=0.0,             # confinement force of CV at boundaries
    equil_temp=300.0,       # equilibrium temperature of simulation
    periodicity=periodicity,
    kinetics=True,          # calculate importent metrics to get accurate kinetics
    verbose=True,           # print verbose output
)
the_md.set_sampling_algorithm(the_bias) # to take affect the sampling algorithm has to be set in the MD interface

nsteps = 5000000
N_reweights = 100
steps_per_iter = int(nsteps/N_reweights)
biaspots = []

for _ in range(N_reweights):
    the_md.run(nsteps=steps_per_iter, traj_file=f"CV_traj.dat")
    biaspots.append(the_bias.metapot[0] * units.atomic_to_kJmol * units.kJ_to_kcal)
    np.save(f"biaspots.npy", biaspots)

the_md.run(nsteps=1, traj_file=f"{this_dir}/CV_traj.dat")
