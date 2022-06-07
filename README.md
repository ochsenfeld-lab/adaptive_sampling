Adaptive Sampling
=================

This package implements various sampling algorithms for the calculation of free energy profiles of molecular transitions. 

## Available sampling methods include:
*	Adaptive Biasing Force (ABF) method [3] 
	
* 	Extended-system ABF (eABF) [4]

	* On-the-fly free energy estimate from the Corrected Z-Averaged Restraint (CZAR) [4]
	
	* Application of Multistate Bannett's Acceptance Ratio (MBAR) [2] to recover full statistical information in post-processing [1]
	
* 	Well-Tempered Metadynamics (WTM) [5] and WTM-eABF [6]

* 	Gaussian-accelerated MD (GaMD) [7] and GaWTM-eABF [8]

## Install:
To install adaptive_sampling download the repository to a local folder and type:
```shell
$ pip install adaptive_sampling
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
    bias_force = eABF.step_bias(write_output=True, write_traj=True)
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

# run MBAR and compute free energy profile and propability density from statistical weights
traj_list, indices, meta_f = mbar.get_windows(grid, cv, la, sigma, equil_temp=300.0)

weights = mbar.run_mbar(traj_list, meta_f, conv=1.0e-4, conv_errvec=None, outfreq=100, equil_temp=300.0)
pmf, rho = mbar.pmf_from_weights(grid, cv[indices], weights, equil_temp=300.0)
```

## Documentation:
Code documentation can be created with pdoc3:
```shell
$ pip install pdoc3
$ pdoc --html adaptive_sampling -o doc/
```
## References:
This work:
1.  Hulm et. al., J. Chem. Phys. (in press)

Implemented methods:

2.  Shirts et. al., J. Chem. Phys. (2008); <https://doi.org/10.1063/1.2978177>
3.  Comer et. al., J. Phys. Chem. B (2015); <https://doi.org/10.1021/jp506633n> 
4.  Lesage et. al., J. Phys. Chem. B (2017); <https://doi.org/10.1021/acs.jpcb.6b10055>
5.  Barducci et. al., Phys. rev. lett. (2008); <https://doi.org/10.1103/PhysRevLett.100.020603>
6.  Fu et. al., J. Phys. Chem. Lett. (2018); <https://doi.org/10.1021/acs.jpclett.8b01994>
7.  Miao et. al., J. Chem. Theory Comput. (2015); <https://doi.org/10.1021/acs.jctc.5b00436>
8.  Chen et. al., J. Chem. Theory Comput. (2021); <https://doi.org/10.1021/acs.jctc.1c00103>
