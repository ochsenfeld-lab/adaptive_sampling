#!/usr/bin/env python
#Imports
import sys
import os
import numpy as np
import random
import time
import subprocess

sys.path.insert(1, '../')
from InterfaceMD import *
from adaptive_biasing_2D import * 

bohr2angs = 0.52917721092e0
################# Imput Section ####################
# MD
seed            = np.random.randint(2147483647) # upper boundary is simply largest signed int value 
nsteps          = 4400000   # number of MD steps
dt              = 5.0e0     # fs
target_temp     = 300.0     # K
mass            = 10.0
friction        = 1.0e-3
coords          = [0.0,0.0]
potential       = '2'

ats_eabf = [[1,-50.0,50.0,2.0,2.0,20.0,6.0]]
grid = True
N_full = 100
height = 1.0
update_int = 20
WT_dT = 4000
f_conf = 500.0

ats_gamd    = [[1, -100, 100, 2]]
equil_steps = 400000
init_steps  = 50000
sigma0      = 3.5
output_freq = 10000

#################### Pre-Loop ####################
start_loop = time.perf_counter()
step_count = 0

the_md = MD(mass_in=mass,coords_in=coords,potential=potential,dt_in=dt,target_temp_in=target_temp,seed_in=seed)

the_bias = ABM(the_md, ats_gamd, method = 'GaMD', output_freq=output_freq, random_seed = seed)
the_eabf = ABM(the_md, ats_eabf, method = 'meta-eABF', output_freq=output_freq, f_conf=f_conf, random_seed = seed)

the_md.calc_init()

the_bias.GaMD(sigma0=sigma0, write_traj=False)
the_eabf.meta_eABF(N_full=N_full, gaussian_height=height, update_int=update_int, WT_dT=WT_dT, grid=grid, write_traj=False)

the_md.calc_etvp()

print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" %(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp))

#################### the MD loop ####################
while step_count < nsteps:
    start_loop = time.perf_counter()
    the_md.step += 1
    step_count  += 1
	
    the_md.propagate(langevin=True, friction=friction)
    the_md.calc()
    
    start = time.perf_counter()
    the_bias.GaMD(sigma0=sigma0, init_steps=init_steps, equil_steps=equil_steps, confine=False, write_traj=False)
    if the_md.step >= equil_steps:
        the_eabf.meta_eABF(N_full=N_full, gaussian_height=height, update_int=update_int, WT_dT=WT_dT, grid=grid, confine=True, write_traj=False)
        if the_md.step >= equil_steps+output_freq:
            the_eabf.write_traj(extended = True, additional_data=the_bias.deltaV)
    t = time.perf_counter() - start
    
    the_md.up_momenta(langevin=True, friction=friction)
    the_md.calc_etvp()

    print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f"%(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp,t))
        
