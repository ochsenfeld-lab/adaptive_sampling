#!/usr/bin/env python
import time
import sys
from adaptive_sampling.sampling_tools.metaeabf import WTMeABF
from adaptive_sampling.interface.interfaceMD_2D import *

bohr2angs = 0.52917721092e0
################# Imput Section ####################
# MD
seed            = 42  
nsteps          = 2000000  # number of MD steps
dt              = 5.0e0     # fs
target_temp     = 300.0     # K
mass            = 10.0
friction        = 1.0e-3
coords          = [-50.0,0.0]
potential       = '2'

# meta-eABF
ats_eabf = [["x", [], -50.0, 50.0, 2.0]] 

ext_sigma = 2.0
ext_mass= 20.0
mtd_std = 6.0
grid = True
N_full = 100
height = 1.0
update_freq = 20
WT_dT = 4000
f_conf = 500.0

#################### Pre-Loop ####################
start_loop = time.perf_counter()
step_count = 0

the_md = MD(
    mass_in=mass,
    coords_in=coords,
    potential=potential,
    dt_in=dt,
    target_temp_in=target_temp,
    seed_in=seed,
)
the_eabf = WTMeABF(
    ext_sigma,
    ext_mass,
    height,
    mtd_std,
    the_md, 
    ats_eabf,
    update_freq=update_freq,
    well_tempered_temp=WT_dT,
    force_on_grid=False,
    friction=friction,
    seed_in=seed,
    output_freq=1000, 
    f_conf=f_conf, 
)

the_md.calc_init()
the_eabf.step_bias()

the_md.calc_etvp()

print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" %(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp))

#################### the MD loop ####################
while step_count < nsteps:
    start_loop = time.perf_counter()
    the_md.step += 1
    step_count  += 1
	
    the_md.propagate(langevin=True, friction=friction)
    the_md.calc()
    
    the_md.forces += the_eabf.step_bias()
    
    the_md.up_momenta(langevin=True, friction=friction)
    the_md.calc_etvp()

    print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"%(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp))
        
