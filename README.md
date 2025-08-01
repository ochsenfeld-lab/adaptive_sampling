Adaptive Sampling
=================

This package implements various sampling algorithms for the calculation of free energy profiles of molecular transitions. 
## Available Sampling Methods Include:
*	Adaptive Biasing Force (ABF) method [1] 
	
* 	Extended-system ABF (eABF) [2]

	* On-the-fly free energy estimate from the Corrected Z-Averaged Restraint (CZAR) [2]
	
	* Application of Multistate Bannett's Acceptance Ratio (MBAR) [3] to recover full statistical information in post-processing [4]
	
* 	(Well-Tempered) Metadynamics (WTM) [5] and WTM-eABF [6]

*   On-the-fly Probability Enhanced Sampling (OPES) [7] and OPES-eABF [8]

* 	Accelerated MD (aMD), Gaussian accelerated MD (GaMD), Sigmoid Accelerated MD (SaMD) [9, 10, 11]

*	Gaussian-accelerated WTM-eABF [12]

*   Free-energy Nudged Elastic Band Method [13]

## Implemented Collective Variables:

*   Distances, angles and torsion angles as well as linear combinations thereof

*   Coordination numbers 

*   Minimized Cartesian RMSD (Kabsch algorithm)

*   Adaptive path collective variables (PCVs) [14, 15]

## Install:
To install adaptive_sampling type:
```shell
$ pip install git+https://github.com/ochsenfeld-lab/adaptive_sampling.git
```
WARNING: The package version that can be installed from PyPI using `pip install adaptive_sampling` is outdated and no longer supported.

## Documentation:
Visit the full documentation at [https://ochsenfeld-lab.github.io/adaptive_sampling/](https://ochsenfeld-lab.github.io/adaptive_sampling/).

## References:
1.  Comer et al., J. Phys. Chem. B (2015); <https://doi.org/10.1021/jp506633n> 
2.  Lesage et al., J. Phys. Chem. B (2017); <https://doi.org/10.1021/acs.jpcb.6b10055>
3.  Shirts et al., J. Chem. Phys. (2008); <https://doi.org/10.1063/1.2978177>
4.  Hulm et al., J. Chem. Phys. (2022); <https://doi.org/10.1063/5.0095554>
5.  Barducci et al., Phys. rev. lett. (2008); <https://doi.org/10.1103/PhysRevLett.100.020603>
6.  Fu et al., J. Phys. Chem. Lett. (2018); <https://doi.org/10.1021/acs.jpclett.8b01994>
7.  Invernizzi et al., J. Phys. Chem. Lett. (2020); <https://doi.org/10.1021/acs.jpclett.0c00497>
8.  Hulm et al., J. Chem. Theory Comput. (2025); <https://doi.org/10.1021/acs.jctc.5c00395>
9.  Hamelberg et al., J. Chem. Phys. (2004); <https://doi.org/10.1063/1.1755656>
10. Miao et al., J. Chem. Theory Comput. (2015); <https://doi.org/10.1021/acs.jctc.5b00436>
11. Zhao et al., J. Phys. Chem. Lett. (2023); <https://doi.org/10.1021/acs.jpclett.2c03688>
12. Chen et al., J. Chem. Theory Comput. (2021); <https://doi.org/10.1021/acs.jctc.1c00103>
13. Semelak et al., J. Chem. Theory Comput. (2023); <https://doi.org/10.1021/acs.jctc.3c00366>
14. Branduardi, et al., J. Chem. Phys. (2007); <https://doi.org/10.1063/1.2432340>
15. Leines et al., Phys. Ref. Lett. (2012); <https://doi.org/10.1103/PhysRevLett.109.020601>

## This and Related Work:
If you use this package in your work please cite:
* 	Hulm et al., J. Chem. Phys., 157, 024110 (2022); <https://doi.org/10.1063/5.0095554>

Other related references:
* 	Hulm et al., J. Chem. Phys., 157, 024110 (2022); <https://doi.org/10.1063/5.0095554>
*	Dietschreit et al., J. Chem. Phys., (2022); <https://aip.scitation.org/doi/10.1063/5.0102075>
*   Hulm et al., J. Chem. Theory. Comput., (2023); <https://doi.org/10.1021/acs.jctc.3c00938>
*   Stan et. al., ACS Cent. Sci., (2024); <https://doi.org/10.1021/acscentsci.3c01403>
*   Pöverlein et. al., J. Chem. Theory, Comput., (2024); <https://doi.org/10.1021/acs.jctc.4c00199>
*   Hulm et al., J. Chem. Theory Comput. (2025); <https://doi.org/10.1021/acs.jctc.5c00395>

