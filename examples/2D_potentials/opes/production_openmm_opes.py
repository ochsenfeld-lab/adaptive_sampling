from sys import stdout
import numpy as np

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmplumed import *

from adaptive_sampling.sampling_tools import *
from adaptive_sampling.interface.interface_openmm import AdaptiveSamplingOpenMM

# ------------------------------------------------------------------------------------
# CONFIGURE PARAMETER
# ------------------------------------------------------------------------------------
config_nsteps = 2e6  # Number of simulation steps
config_explore = False  # Enable Exploration mode
config_adaptive_sigma = True  # Enable adaptive sigma calculation according to welford
config_unbiased_time = (
    10  # If no input sigma this many update freq times unbiased steps are simulated
)
config_input = False  # If False enables unbiased estimation of sigma 0
config_fixed_sigma = (
    False  # Disable bandwidth rescaling and use input sigma for all kernels
)
config_merge = 1.0  # Merging or not, np.inf is no merging
config_update_freq = 500  # Frequency in which kernels are placed
config_recursion_merge = True  # Enable recursive merging
config_approx_norm_factor = True  # Enable approximation of norm factor
config_md_print_freq = 10000  # Print frequency in .grid file
config_f_conf = 1000.0  # Confinment force to keep system in boundaries
config_bias_factor = None  # Direct setting of bias factor, if calculation from energy is wanted put in 'None'
config_energy_barr = 70.0  # Energy barrier to overcome
config_nsteps_output = 5000  # Frequency in which outfile for postprocessing is written
config_verbose = False  # Enable for debugging
# ------------------------------------------------------------------------------------

####################################################################################
# Input parameters...
####################################################################################
this_dir = os.environ["THIS_DIR"]

# input and output files
# pdb = PDBFile('PUS.comb.shaw.mg.salted.pdb')
prm = AmberPrmtopFile(f"{this_dir}/PUS_new.parm7")
crd = AmberInpcrdFile(f"{this_dir}/PUS_new.rst7")

start_from_chk = True
chk_file = f"{this_dir}/Production_restart.chk"
out_name = "OPES_production"

save_chk = True
chk_freq = 10000
save_dcd = True
dcd_freq = 1000
save_report = True
report_freq = 1000

# MD properties
target_temperature = 310.0 * kelvin
add_barostat = True
target_pressure = 1.0 * bar
time_step = 2.0 * femtoseconds
production_steps = config_nsteps

platform = "CPU"
# properties = {'Precision': 'mixed', "DeviceIndex": "0,1,2,3"}

####################################################################################
# Preparing system and integrator...
####################################################################################
system = prm.createSystem(
    nonbondedMethod=PME,
    nonbondedCutoff=1.2 * nanometer,
    constraints=HBonds,
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
)

# add Barostat
if add_barostat:
    print("Adding Monte Carlo Barostat to system")
    barostat = MonteCarloBarostat(target_pressure, target_temperature)
    system.addForce(barostat)

# ------------------------------------------------------------------------------------
# define collective variables
cv_atoms = [2828, 1194]  # Arg-Asp distance
minimum = 2.0  # minimum of the CV
maximum = 20.0  # maximum of the CV
bin_width = 0.1  # bin with along the CV

collective_var = [
    ["distance", cv_atoms, minimum, maximum, bin_width],
]

periodicity = [None]

# Initialize the `AdaptiveSamplingOpenMM` interface to couple the OpenMM simulaiton to an bias potential
# the Openmm `simulation` object is set up internally, but can still be modified by calling `the_md.simulation`
the_md = AdaptiveSamplingOpenMM(
    crd.positions,
    prm.topology,
    system,
    dt=time_step,  # timestep in fs
    equil_temp=target_temperature,  # temperature of simulation
    langevin_damping=1.0 / picosecond,  # langevin damping in 1/ps
    calc_energy=True,  # if energy should be returned every step, this may slow down simulations
    cv_atoms=cv_atoms,  # specifying CV atoms significantly speeds up simulation of large systems, as the bias force will only be calculated for those
)
the_md.integrator.setConstraintTolerance(0.00001)

if start_from_chk:
    the_md.restart(chk_file)

# Append OpenMM reporters to simulation for output
if save_dcd:
    the_md.simulation.reporters.append(DCDReporter(f"{out_name}.dcd", dcd_freq))
if save_chk:
    the_md.simulation.reporters.append(
        CheckpointReporter(f"{this_dir}/{out_name}_restart.chk", chk_freq)
    )
if save_report:
    the_md.simulation.reporters.append(
        StateDataReporter(
            stdout,
            report_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            progress=False,
            remainingTime=False,
            speed=True,
            totalSteps=production_steps,
            separator="\t",
        )
    )

# --------------------------------------------------------------------------------------
# Setup the sampling algorithm
output_freq = 1000  # frequency of writing outputs
kernel_std = (
    np.array([1.05]) if config_input else np.array([None])
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
    verbose=config_verbose,
    recursion_merge=config_recursion_merge,
    update_freq=config_update_freq,
    f_conf=config_f_conf,
    bias_factor=config_bias_factor,
)


the_md.set_sampling_algorithm(
    the_bias
)  # to take affect the sampling algorithm has to set in the MD interface

the_md.run(nsteps=config_nsteps)  # 500000 * 2 fs = 1 ns

lastpositions = the_md.simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(
    prm.topology, lastpositions, open(f"{this_dir}/{out_name}_final.pdb", "w")
)

#######################################################################################
print("Done!")
