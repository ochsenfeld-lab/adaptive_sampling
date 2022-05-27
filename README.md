# Adaptive Sampling

This package implements various algorithms for enhanced sampling of molecular transitions. 

## Available sampling methods include:
*	Adaptive Biasing Force (ABF) method [3] 
	
* 	Extended-system ABF (eABF) [4]

	On-the-fly free energy estimate from the Corrected z-Averaged Restraint (CZAR) [4]
	
	Application of Multistate Bannet's Acceptance Ration (MBAR) [2] to recover full statistical information in post-processing [1]
	
* 	Well-Tempered Metadynamics (WTM) [5] and WTM-eABF [6]

* 	Gaussian-accelerated MD (GaMD) [7] and GaWTM-eABF [8]

## Install:
> $ pip install adaptive_sampling 

## Requirements:
* python >= 3.8
* numpy >= 1.19
* torch >= 1.10
* scipy >= 1.8

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