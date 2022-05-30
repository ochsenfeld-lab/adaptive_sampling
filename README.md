# Adaptive Sampling

This package implements various sampling algorithms for the calculation of free energy profiles of molecular transitions. 

### Available sampling methods include:
*	Adaptive Biasing Force (ABF) method [3] 
	
* 	Extended-system ABF (eABF) [4]

	* On-the-fly free energy estimate from the Corrected z-Averaged Restraint (CZAR) [4]
	
	* Application of Multistate Bannet's Acceptance Ration (MBAR) [2] to recover full statistical information in post-processing [1]
	
* 	Well-Tempered Metadynamics (WTM) [5] and WTM-eABF [6]

* 	Gaussian-accelerated MD (GaMD) [7] and GaWTM-eABF [8]

## Install:
> $ pip install adaptive_sampling 

## Requirements:
* python >= 3.8
* numpy >= 1.19
* torch >= 1.10
* scipy >= 1.8

## Basic usage:
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
the_md = MD(...)
collective_var = ["distance", list_of_atom_indices, minimum, maximum, bin_width]
the_bias = eABF(ext_sigma, ext_mass, the_md, collective_variable, output_freq=10, f_conf=100, equil_temp=300.0)
the_md.forces += eABF.step_bias(write_output=True, write_traj=True)
```
This automatically writes an on-the-fly free energy estimate in the output file and all necessary data for post-processing in a trajectory file.
For extended-system dynamics unbiased statistical weigths of individual frames can be obtained using the MBAR estimator:
```python
import numpy as np
from adaptive_sampling.processing_tools import mbar

traj_dat = np.loadtxt('CV_traj.dat', skiprows=1)
ext_sigma = 2.0
bin_width = 2.0
grid = np.arange(70.0, 170.0, bin_width)

cv = traj_dat[:,1]  # trajectory of collective variable
la = traj_dat[:,2]  # trajectory of extended system

traj_list, indices, meta_f = mbar.get_windows(grid, cv, la, sigma, equil_temp=300.0)

weigths = mbar.run_mbar(traj_list, meta_f, conv=1.0e-4, conv_errvec=None, outfreq=100, equil_temp=300.0)
pmf, rho = mbar.pmf_from_weights(grid, cv[indices], W, equil_temp=300.0)
```

## Documentation:
Code documentation can be created with pdoc3:
```shell
$ pip install pdoc3

$ pdoc --html adaptive_sampling -o doc/
```
## References:
This work:
1. 	Hulm et. al., J. Chem. Phys. (in press)

Implemented methods:

2.  Shirts et. al., J. Chem. Phys. (2008); <https://doi.org/10.1063/1.2978177>
3.	Comer et. al., J. Phys. Chem. B (2015); <https://doi.org/10.1021/jp506633n> 
4.  Lesage et. al., J. Phys. Chem. B (2017); <https://doi.org/10.1021/acs.jpcb.6b10055>
5.  Barducci et. al., Phys. rev. lett. (2008); <https://doi.org/10.1103/PhysRevLett.100.020603>
6.  Fu et. al., J. Phys. Chem. Lett. (2018); <https://doi.org/10.1021/acs.jpclett.8b01994>
7.  Miao et. al., J. Chem. Theory Comput. (2015); <https://doi.org/10.1021/acs.jctc.5b00436>
8.  Chen et. al., J. Chem. Theory Comput. (2021); <https://doi.org/10.1021/acs.jctc.1c00103>
