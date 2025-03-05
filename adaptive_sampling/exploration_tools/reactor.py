#!/usr/bin/env python3
import sys, os
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from ..interface.sampling_data import MDInterface
from ..units import *
from .utils import filter_and_index_bo


class Reactor(ABC):
    """Abstract class for molecular dynamics based reaction space exploration using reactor methods

    Args:
        md: Object of the MD Interface
        init_temp: initial temperature of MD
        equil_temp: equilibrium temperature of MD
        verbose: print verbose information
        output_freq: frequency in steps for writing outputs
    """
    def __init__(
        self,
        r_min: float,
        r_max: float,
        md: MDInterface,
        verbose: bool = True,
        #mols_array = [],
        *args,
        **kwargs,
    ):
        self.the_md = md
        self.verbose = verbose

        self.r_max = r_max/BOHR_to_ANGSTROM
        self.r_min = r_min/BOHR_to_ANGSTROM
     
        
    @abstractmethod
    def step_bias(self):
        pass

    def write_pop_output(self, prefix: str = "aseMD", pop_array: np.array = None):
        """Print Mulliken population analysis to file
        
        Args:
            prefix: prefix for filename of output file
        """
        try:
            pop_old = np.load(prefix + "_pop.npz", allow_pickle=True)
            pop_combined = np.concatenate((pop_old, pop_array), axis=0)
        except FileNotFoundError:
            pop_combined = pop_array
        
        np.savez_compressed(prefix + "_pop.npz", pop_combined)

    def write_bond_order_output(self, prefix: str = "aseMD", bo_array: np.array = None):
        """saves bond orders in a ij order for each time step (needed for the post-processing)

        Args:
            filename: name of bond-orders file
            bo: matrix form of Wiberg/Mayer bond orders from the calculator
        """
        bo_array_filtered = np.array([bo_array[0], filter_and_index_bo(bo_array[1])])
        try:
            bo_old = np.load(prefix + "_pop.npz", allow_pickle=True)
            bo_combined = np.concatenate((bo_old, bo_array_filtered), axis=0)
        except FileNotFoundError:
            bo_combined = bo_array_filtered
        
        np.savez_compressed(prefix + "_bo.npz", bo_combined)
'''

    def write_bond_order_output(self, prefix: str = "aseMD", bo_list: list = None):
        """saves bond orders in a ij order for each time step (needed for the post-processing)

        Args:
            filename: name of bond-orders file
            bo: matrix form of Wiberg/Mayer bond orders from the calculator
        """
        md_state = self.the_md.get_sampling_data()

        step = md_state.step
        dt = md_state.dt
        natoms = md_state.natoms

        if bo.all() == None:
            print("Bond order matrix was not provided!")
        else:


            with open(prefix + "_bo.dat", "a+") as f:
                string = str("TIME: %14.7f\n") % (step*dt)

                f.write(string)
                for i in range(0,natoms):
                    for j in range(i+1,natoms):
                        string = str("%20.10e") % (bo[i][j])
                        f.write(string)
                        f.write("\n")
                f.write("\n")
                f.close()
'''



