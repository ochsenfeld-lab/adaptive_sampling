Basic usage
===========

This section provides a brief overview of how to use the `adaptive_sampling` package. For more detailed information, please refer to the tutorials and code documentation.

Molecular dynamics (MD) interfaces
----------------------------------
The `adaptive_sampling` package provides interfaces to various molecular dynamics engines, including OpenMM, ASE, and ASH. 
These interfaces allow you to run simulations using the adaptive sampling methods provided by the package.
 * **AseMD**: For running *ab initio* molecular dynamics simulations using the calculators from the atomic simulation environment (ASE), which provide interfaces to various quantum chemistry packages. 
 * **AshMD**: For running *ab initio* molecular dynamics simulations using the ASH package, which provides many quantum chemistry packages and additionally supports QM/MM simulations.
 * **AdaptiveSamplingOpenMM**: Allows performing molecular dynamics simulations using the OpenMM engine. 
 * **InterfaceMD_2D**: Provides 2D potentials for testing the adaptive sampling methods.

The ASE interface
^^^^^^^^^^^^^^^^^

The `AseMD` class allows for running adaptive sampling algorithms using any ASE calculator.
Refer to the ASE `documentation <https://wiki.fysik.dtu.dk/ase/>`_ for more information on available calculators and molecular modeling workflows in ASE.
In the below example we will use the `tblite` package and the GFN2-xTB tight-binding method. 

.. code-block:: python
    :linenos:
    
    # create an ASE fragment from an XYZ files
    from ase.io import read
    frag = read("structure.xyz")

    # attach the tblite calculator to the fragment
    from tblite.ase import TBLite
    frag.calc = TBLite(method="GFN2-xTB")

    # initilize the AseMD object
    from adaptive_sampling.interface import AseMD
    the_md = AseMD(
        atoms=frag,               # ASE fragment 
        dt=1.0,                   # Timestep in fs
        thermostat=True,          # Apply an langevin thermostat
        target_temp=300.0,        # The thermostat temperature
        scratch_dir="./scratch/", # Location where scratch files of the calculator should be written
    )

    # initilize one or multiple sampling methods from `sampling_tools`
    from adaptive_sampling.sampling_tools import *
    the_bias = ...

    # Initialize the MD
    the_md.calc_init(
        init_momenta="random",   # Method for momenta initialization
        biaspots=[the_bias, ],   # Set the sampling algorithms
        init_temp=300.0,         # The initial temperature, if `init_momenta="random"`
        restart_file=None,       # Filename of restart file, if `init_momenta="read"`
    )

    # Ready for running the MD
    the_md.run(nsteps=10000)

The ASH interface
^^^^^^^^^^^^^^^^^

The `AshMD` class allows for running adaptive sampling algorithms using any ASH theory object.
Refer to the ASH `documentation <https://ash.readthedocs.io/en/latest/>`_ for more information on available theory objects and molecular modeling workflows in ASH.
In the below example we will perform a QM/MM MD simulation using the xTB method and an AMBER force field. 

.. code-block:: python
    :linenos:

    # Create the ASH fragment from a PDB
    import ash
    frag = ash.Fragment(pdbfile=f"structure.pdb")

    # Create the MM Theory
    mm_theory = ash.OpenMMTheory(
        cluster_fragment=frag,               # ASH Fragment 
        Amberfiles=True,                     # Use Amber parameter file
        amberprmtopfile=f"{mm_path}.parm7",  # The Amber parameter file
        periodic=True,                       # Periodic boundary conditions or not.
    )

    # Create the QM Theory
    qm_theory = ash.xTBTheory(
        xtbmethod="GFN2",           # The xTB method
        runmode="inputfile",        # Only "inputfile" supports for QM/MM
    )

    # Create the QM/MM System
    qm_atoms = [i for i in range(0, 23)] # Indices of QM atoms
    qmmm_theory = ash.QMMMTheory(
        qm_theory=qm_theory,        # ASH QM Theory object
        mm_theory=mm_theory,        # ASH MM Theory object (should be OpenMMTheory)
        fragment=frag,              # ASH Fragment
        embedding="Elstat",         # QM/MM embedding type
        qmatoms=qm_atoms,           # The QM atoms (list of atom indices)
    )

    # Initialize the AseMD interface
    from adaptive_sampling.interface.interfaceASH import AshMD
    the_md = AshMD(
        fragment=frag,              # ASH fragment
        calculator=qmmm_theory,     # ASH calculator 
        dt=1.0,                     # Time step in fs
        thermostat=True,            # Apply Langevin thermostat 
        target_temp=300.0,          # The target temperature in Kelvin
        barostat=True,              # Apply Monte-Carlo barostat
        target_pressure=1.0,        # The target pressure in Bar
        barostat_freq=25,           # Frequency of updating the barostat
    )

    # Initilize one or multiple sampling methods from `sampling_tools`
    from adaptive_sampling.sampling_tools import *
    the_bias = ...

    # Initialize the MD
    the_md.calc_init(
        init_momenta="random",      # Method for momenta initialization
        biaspots=[the_bias, ],      # Set the sampling algorithms
        init_temp=300.0,            # The initial temperature, if `init_momenta="random"`
        restart_file=None,          # Filename of restart file, if `init_momenta="read"`
    )

    # Ready for running the MD
    the_md.run(nsteps=1000)

The OpenMM interface
^^^^^^^^^^^^^^^^^^^^

This section provides a minimal example for using the OpenMM interface together with AMBER style parameter files.
You can initialize the `AdaptiveSamplingOpenMM` class as follows:

.. code-block:: python
    :linenos:

    from sys import stdout

    from openmm import *
    from openmm.app import *
    from openmm.unit import *

    from adaptive_sampling.interface import AdaptiveSamplingOpenMM

    # Setup OpenMM
    prmtop = AmberPrmtopFile(f"../data/alanine-dipeptide.prmtop")
    crd = AmberInpcrdFile(f"../data/alanine-dipeptide.crd")
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    # Initialize the `AdaptiveSamplingOpenMM` interface to couple the OpenMM simulation to a bias potential
    the_md = AdaptiveSamplingOpenMM(
        crd.positions,
        prmtop.topology,
        system,
        dt=2.0,               # timestep in fs
        equil_temp=300.0,     # temperature of simulation in Kelvin
        langevin_damping=1.0, # langevin damping in 1/ps
    )

    # The OpenMM `simulation` and `integrator` objects are set up internally, but can still be modified by calling `the_md.simulation` or `the_md.integrator`
    the_md.integrator.setConstraintTolerance(0.00001)
    the_md.simulation.reporters.append(DCDReporter('alanine-dipeptide.dcd', 100)) 

Before running the MD any importance sampling algorithm from `sampling_tools` has to be attached to the OpenMM interface:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import *
    the_bias = ... # init sampling algorithm
    the_md.set_sampling_algorithm(the_bias) 

If you want to apply multiple sampling algorithms, you can specify those as a list:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import *
    the_bias1 = ... # init first sampling algorithm
    the_bias2 = ... # init second sampling algorithm, e.g. additional harmonic constraint
    the_md.set_sampling_algorithm([the_bias1, the_bias2]) 

Now, the MD is ready to run:

.. code-block:: python
    :linenos:

    the_md.run(nsteps=500000) # 500000 * 2 fs = 1 ns


Importance sampling
-------------------

Importance algorithms facilitate the calculation of reaction and activation free energies by sampling molecular transitions.
In the `adaptive sampling` package these are located in the `sampling_tools` subpackage.

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import *

Implemented is a wide range of sampling algorithms, including well-tempered metadynamics (WTM) and its successor on-the-fly probability enhanced sampling (OPES), the adaptive biasing force (ABF) method or extended-system based hybrid methods (WTM-eABF, OPES-eABF).

Exploration tools
-----------------

