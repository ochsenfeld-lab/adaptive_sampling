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

    def _write_traj(self, data: dict = {}, filename: str = "CV_traj.dat"):
        """write trajectory of extended or normal ABF at output times

        Args:
            data: data to write
            filename: name of trajectory file
        """
        step = self.the_md.get_sampling_data().step

        # write header
        if not os.path.isfile(filename) and step == 0:
            # start new file in first step

            with open(filename, "w") as traj_out:
                traj_out.write("%14s\t" % "time [fs]")
                for i in range(len(self.traj[0])):
                    traj_out.write("%14s\t" % f"Xi{i}")
                for kw in data.keys():
                    traj_out.write("%14s\t" % kw)
                if self.kinetics:
                    traj_out.write("%14s\t" % "m_xi_inv [a.u.]")
                    traj_out.write("%14s\t" % "|dU| [a.u.]")
                    traj_out.write("%14s\t" % "|dxi| [a.u.]")
                    traj_out.write("%14s\t" % "dU*dxi [a.u.]")

        elif step > 0:
            # append new steps of trajectory since last output
            with open(filename, "a") as traj_out:
                for n in range(self.out_freq):
                    traj_out.write(
                        "\n%14.6f\t"
                        % (
                            (step - self.out_freq + n)
                            * self.the_md.get_sampling_data().dt
                        )
                    )  # time in fs
                    for i in range(len(self.traj[0])):
                        if self.cv_type[i] == "angle":
                            traj_out.write(
                                "%14.6f\t"
                                % (
                                    self.traj[-self.out_freq + n][i]
                                    * DEGREES_per_RADIAN
                                )
                            )
                        elif self.cv_type[i] == "distance":
                            traj_out.write(
                                "%14.6f\t"
                                % (self.traj[-self.out_freq + n][i] * BOHR_to_ANGSTROM)
                            )
                        else:
                            traj_out.write(
                                "%14.6f\t" % (self.traj[-self.out_freq + n][i])
                            )

                    for val in data.values():
                        traj_out.write("%14.6f\t" % (val[-self.out_freq + n]))

                    if self.kinetics:
                        traj_out.write(
                            "%14.6f\t" % (self.mass_traj[-self.out_freq + n])
                        )
                        traj_out.write(
                            "%14.6f\t" % (self.abs_forces[-self.out_freq + n])
                        )
                        traj_out.write(
                            "%14.6f\t" % (self.abs_grad_xi[-self.out_freq + n])
                        )
                        traj_out.write(
                            "%14.6f\t" % (self.CV_crit_traj[-self.out_freq + n])
                        )

    def _write_restart(self, filename, **kwargs):
        """save **kwargs in .npz file"""
        np.savez(filename, **kwargs)
        sys.stdout.flush()