Introduction
============

This page provides an introduction to the `adaptive_sampling` package, which is designed to facilitate reaction space exploration and 
importance sampling in molecular dynamics simulations.
The package includes the following submodules:

 * **exploration_tools**: Provides tools for exploring the chemical space of molecular systems using the ab initio Nanoreactor or Hyperreactor dynamics. 
 * **sampling_tools**: Contains tools for importance sampling of molecular transitions, facilitating the calculation associated of reaction and activation free energies. 
 * **processing_tools**: Provides tools for processing and analyzing simulation data, including methods for the automatic generation of reaction networks from exploration runs and methods to calculate unbiased ensample properties from importance sampling runs.
 * **interface**: Contains interfaces for integrating with molecular dynamics engines. Available are interfaces to `OpenMM <https://openmm.org/>`_ (for MM simulations), `ASE <https://wiki.fysik.dtu.dk/ase/>`_ (for QM simulations) and `ASH <https://ash.readthedocs.io/en/latest/>`_ (for QM/MM simulations).
 * **colvars**: Provides a large set of collective variable (CVs) for the sampling tools, including common CVs like distance, angle, and dihedrals, and more complex ones like path CVs and machine learning CVs.


