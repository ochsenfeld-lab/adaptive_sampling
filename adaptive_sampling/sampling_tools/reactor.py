#!/usr/bin/env python3
import sys, os
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from ..interface.sampling_data import MDInterface
from .utils import diff
from ..units import *


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
        md: MDInterface,
        verbose: bool = True,
        output_freq: int = 50,
        **kwargs,
    ):
        self.the_md = md
        self.out_freq = output_freq
        self.verbose = verbose

        # store trajectories of CVs and temperature and epot between outputs
        md_state = self.the_md.get_sampling_data()
        self.temp = [md_state.temp]
        self.epot = [md_state.epot]

    @abstractmethod
    def step_bias(self):
        pass

    #@abstractmethod
    #def write_restart(self):
    #    pass

    #@abstractmethod
    #def restart(self):
    #    pass

    #@abstractmethod
    #def write_traj(self):
    #    pass
    
    
    def write_output(self, data: dict, filename="free_energy.dat"):
        """write results to output file

        Args:
            data: results to write
            filename: name of output file
        """
        grid = np.copy(self.grid)
        for i in range(self.ncoords):
            if self.the_cv.type == "angle":
                grid *= DEGREES_per_RADIAN
            elif self.the_cv.type == "distance":
                grid *= BOHR_to_ANGSTROM

        # head of data columns
        with open(filename, "w") as fout:
            for i in range(self.ncoords):
                fout.write("%14s\t" % "CV{dim}".format(dim=i))
            for kw in data.keys():
                fout.write("%14s\t" % kw)
            fout.write("\n")

            # write data to columns
            if self.ncoords == 1:
                for i in range(self.nbins):
                    fout.write("%14.6f\t" % grid[0][i])
                    for dat in data.values():
                        fout.write("%14.6f\t" % dat[0][i])
                    fout.write("\n")

            if self.ncoords == 2:
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        fout.write("%14.6f\t%14.6f\t" % (grid[1][i], grid[0][j]))
                        for dat in data.values():
                            fout.write("%14.6f\t" % dat[i, j])
                        fout.write("\n")

   