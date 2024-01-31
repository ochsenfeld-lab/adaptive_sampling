Adaptive Sampling
=================

This package implements various sampling algorithms for the calculation of free energy profiles of molecular transitions. 

## Available Sampling Methods Include:
*	Adaptive Biasing Force (ABF) method [1] 
	
* 	Extended-system ABF (eABF) [2]

	* On-the-fly free energy estimate from the Corrected Z-Averaged Restraint (CZAR) [2]
	
	* Application of Multistate Bannett's Acceptance Ratio (MBAR) [3] to recover full statistical information in post-processing [4]
	
* 	(Well-Tempered) Metadynamics (WTM) [5] and WTM-eABF [6]

* 	Accelerated MD (aMD), Gaussian accelerated MD (GaMD), Sigmoid Accelerated MD (SaMD) [7, 8, 9]

*	Gaussian-accelerated WTM-eABF [10]

*   Free-energy Nudged Elastic Band Method [11]

## Implemented Collective Variables:

*   Distances, angles and torsion angles as well as linear combinations thereof

*   Coordination numbers 

*   Minimized Cartesian RMSD (Kabsch algorithm)

*   Adaptive path collective variables (PCVs) [12, 13]

## Install:
To install adaptive_sampling type:
```shell
$ pip install adaptive-sampling
```


## Requirements:
* python >= 3.8
* numpy >= 1.19
* torch >= 1.10
* scipy >= 1.7

## Basic Usage:
To use adaptive sampling with your MD code of choice add a function called `get_sampling_data()` to the corresponding python interface that returns an object containing all required data. Hard-coded dependencies can be avoided by wrapping the `adaptive_sampling` import in a `try/except` clause:

```python
class MD:
    # Your MD code
    ...

    def get_sampling_data(self):
        try:
            from adaptive_sampling.interface.sampling_data import SamplingData

            mass   = ...
            coords = ...
            forces = ...
            epot   = ...
            temp   = ...
            natoms = ...
            step   = ...
            dt     = ...

            return SamplingData(mass, coords, forces, epot, temp, natoms, step, dt)
        except ImportError as e:
            raise NotImplementedError("`get_sampling_data()` is missing `adaptive_sampling` package") from e
```
The bias force on atoms in the N-th step can be obtained by calling `step_bias()` on any sampling algorithm:
```python
from adaptive_sampling.sampling_tools import *

# initialize MD code
the_md = MD(...)

# collective variable
atom_indices = [0, 1] 
minimum   = 1.0  # Angstrom
maximum   = 3.5  # Angstrom
bin_width = 0.1  # Angstrom 
collective_var = [["distance", atom_indices, minimum, maximum, bin_width]]

# extended-system eABF 
ext_sigma = 0.1  # thermal width of coupling between CV and extended variable in Angstrom
ext_mass = 20.0  # mass of extended variable 
the_bias = eABF(
    ext_sigma, 
    ext_mass, 
    the_md, 
    collective_var, 
    output_freq=10, 
    f_conf=100, 
    equil_temp=300.0
)

for md_step in range(steps):
    # propagate langevin dynamics and calc forces 
    ... 
    bias_force = the_bias.step_bias(write_output=True, write_traj=True)
    the_md.forces += bias_force
    ...
    # finish md_step
```
This automatically writes an on-the-fly free energy estimate in the output file and all necessary data for post-processing in a trajectory file.
For extended-system dynamics unbiased statistical weights of individual frames can be obtained using the MBAR estimator:
```python
import numpy as np
from adaptive_sampling.processing_tools import mbar

traj_dat = np.loadtxt('CV_traj.dat', skiprows=1)
ext_sigma = 0.1    # thermal width of coupling between CV and extended variable 

# grid for free energy profile can be different than during sampling
minimum   = 1.0     
maximum   = 3.5    
bin_width = 0.1    
grid = np.arange(minimum, maximum, bin_width)

cv = traj_dat[:,1]  # trajectory of collective variable
la = traj_dat[:,2]  # trajectory of extended system

# run MBAR and compute free energy profile and probability density from statistical weights
traj_list, indices, meta_f = mbar.get_windows(grid, cv, la, ext_sigma, equil_temp=300.0)

exp_U, frames_per_traj = mbar.build_boltzmann(
    traj_list, 
    meta_f, 
    equil_temp=300.0,
)

weights = mbar.run_mbar(
    exp_U,
    frames_per_traj,
    max_iter=10000,
    conv=1.0e-7,
    conv_errvec=1.0,
    outfreq=100,
    device='cpu',
)

pmf, rho = mbar.pmf_from_weights(grid, cv[indices], weights, equil_temp=300.0)
```

## Documentation:
Code documentation can be created with pdoc3:
```shell
$ pip install pdoc3
$ pdoc --html adaptive_sampling -o doc/
```
## References:
1.  Comer et al., J. Phys. Chem. B (2015); <https://doi.org/10.1021/jp506633n> 
2.  Lesage et al., J. Phys. Chem. B (2017); <https://doi.org/10.1021/acs.jpcb.6b10055>
3.  Shirts et al., J. Chem. Phys. (2008); <https://doi.org/10.1063/1.2978177>
4.  Hulm et al., J. Chem. Phys. (2022); <https://doi.org/10.1063/5.0095554>
5.  Barducci et al., Phys. rev. lett. (2008); <https://doi.org/10.1103/PhysRevLett.100.020603>
6.  Fu et al., J. Phys. Chem. Lett. (2018); <https://doi.org/10.1021/acs.jpclett.8b01994>
7.  Hamelberg et al., J. Chem. Phys. (2004); <https://doi.org/10.1063/1.1755656>
8.  Miao et al., J. Chem. Theory Comput. (2015); <https://doi.org/10.1021/acs.jctc.5b00436>
9.  Zhao et al., J. Phys. Chem. Lett. (2023); <https://doi.org/10.1021/acs.jpclett.2c03688>
10.  Chen et al., J. Chem. Theory Comput. (2021); <https://doi.org/10.1021/acs.jctc.1c00103>
11.  Semelak et al., J. Chem. Theory Comput. (2023); <https://doi.org/10.1021/acs.jctc.3c00366>
12.  Branduardi, et al., J. Chem. Phys. (2007); <https://doi.org/10.1063/1.2432340>
13.  Leines et al., Phys. Ref. Lett. (2012); <https://doi.org/10.1103/PhysRevLett.109.020601>

## This and Related Work:
If you use this package in your work please cite:
* 	Hulm et al., J. Chem. Phys., 157, 024110 (2022); <https://doi.org/10.1063/5.0095554>

Other related references:
*	Dietschreit et al., J. Chem. Phys., (2022); <https://aip.scitation.org/doi/10.1063/5.0102075>
*   Hulm et al., J. Chem. Theory. Comput., (2023); <https://doi.org/10.1021/acs.jctc.3c00938>
*   Stan et. al., ACS Cent. Sci., (2024); <https://doi.org/10.1021/acscentsci.3c01403>