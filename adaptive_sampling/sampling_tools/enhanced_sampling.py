#!/usr/bin/env python
import os
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from ..interface.sampling_data import MDInterface
from ..colvars.colvars import CV
from .utils import diff


class EnhancedSampling(ABC):
    """Abstract class for sampling algorithms"""

    def __init__(
        self,
        md: MDInterface,
        cv_def: list,
        equil_temp: float,
        verbose: bool = True,
        kinetics: bool = False,
        f_conf: float = 100,
        output_freq: int = 100,
    ):

        self.the_md = md
        self.the_cv = CV(self.the_md, requires_grad=True)
        self.out_freq = output_freq
        self.equil_temp = equil_temp
        self.verbose = verbose

        # definition of CVs
        self.ncoords = len(cv_def)
        self.cv = np.array([item[0] for item in cv_def], dtype=str)
        self.atoms = [cv_def[i][1] for i in range(self.ncoords)]
        self.minx = np.array([item[2] for item in cv_def], dtype=float)
        self.maxx = np.array([item[3] for item in cv_def], dtype=float)
        self.dx = np.array([item[4] for item in cv_def], dtype=float)
        self.f_conf = np.array([f_conf for _ in range(self.ncoords)], dtype=float)

        self.cv_type = ["" for _ in range(self.ncoords)]
        (xi, delta_xi) = self.get_cv()

        # unit conversion
        self.minx, self.maxx, self.dx = self.unit_conversion_cv(
            self.minx, self.maxx, self.dx
        )
        self.f_conf = self.unit_conversion_force(self.f_conf)

        # store trajectories of CVs and temperature and epot between outputs
        self.traj = np.array([xi])
        self.temp = [self.the_md.get_sampling_data().temp]
        self.epot = [self.the_md.get_sampling_data().epot]

        # get number of bins (1D or 2D)
        self.nbins_per_dim = np.array([1, 1])
        self.grid = []
        for i in range(self.ncoords):
            self.nbins_per_dim[i] = (
                int(np.ceil(np.abs(self.maxx[i] - self.minx[i]) / self.dx[i]))
                if self.dx[i] > 0
                else 1
            )
            self.grid.append(
                np.linspace(
                    self.minx[i] + self.dx[i] / 2,
                    self.maxx[i] - self.dx[i] / 2,
                    self.nbins_per_dim[i],
                )
            )
        self.nbins = np.prod(self.nbins_per_dim)

        # accumulators and conditional averages
        self.bias = np.zeros(
            (self.ncoords, self.nbins_per_dim[1], self.nbins_per_dim[0]), dtype=float
        )
        self.cv_crit = np.copy(self.bias)

        self.histogram = np.zeros(
            (self.nbins_per_dim[1], self.nbins_per_dim[0]), dtype=float
        )
        self.pmf = np.copy(self.histogram)

        # trajectory of mass of CV for postprocessing of kinetics
        self.kinetics = kinetics
        if self.kinetics and self.ncoords == 0:
            self.mass_traj = [self._get_mass_of_cv(delta_xi)]
            self.abs_forces = [np.linalg.norm(self.the_md.forces)]
            self.CV_crit_traj = [np.abs(np.dot(self.the_md.forces, delta_xi[0]))]
            self.abs_grad_xi = [np.linalg.norm(delta_xi)]
        elif self.kinetics:
            self.kinetics = False
            if verbose:
                print(
                    " >>> Warning: kinetics only available for 1D collective variables"
                )

        if self.verbose:
            for i in range(self.ncoords):
                print(f"\n Initialize {self.cv[i]} as collective variable:")
                if self.cv_type[i] == "angle":
                    output_dat = (
                        self.minx[i] / np.pi / 180.0, 
                        self.maxx[i] / np.pi / 180.0, 
                        self.dx[i] / np.pi / 180.0, 
                        'degree'
                    )
                elif self.cv_type[i] == "distance":
                    output_dat = (
                        self.minx[i]  * 0.52917721092e0, 
                        self.maxx[i]  * 0.52917721092e0, 
                        self.dx[i] * 0.52917721092e0,
                        'Angstrom'
                    )
                else:
                    output_dat = (
                        self.minx[i], 
                        self.maxx[i], 
                        self.dx[i],
                        ''
                    )
                print(f"\t Minimum{i}:\t\t\t{output_dat[0]} {output_dat[3]}")
                print(f"\t Maximum{i}:\t\t\t{output_dat[1]} {output_dat[3]}")
                print(f"\t Bin width{i}:\t\t\t{output_dat[2]} {output_dat[3]}")
            print("\t--------------------------------------")
            print(f"\t Total number of bins:\t\t{self.nbins}\n")

    @abstractmethod
    def step_bias(self):
        pass

    @abstractmethod
    def get_pmf(self):
        pass

    @abstractmethod
    def shared_bias(self):
        pass

    def harmonic_walls(self, xi: np.ndarray, delta_xi: np.ndarray) -> np.ndarray:
        """confine system with harmonic walls to range(self.minx, self.maxx)

        args:
            xi: collective variable
            delta_xi: gradient of collective variable

        returns:
            bias_force:
        """
        bias_force = np.zeros_like(self.the_md.forces.ravel())

        for i in range(self.ncoords):
            if xi[i] > self.maxx[i]:
                r = diff(self.maxx[i], xi[i], self.cv_type[i])
                bias_force -= self.f_conf[i] * r * delta_xi[i]

            elif xi[i] < self.minx[i]:
                r = diff(self.minx[i], xi[i], self.cv_type[i])
                bias_force -= self.f_conf[i] * r * delta_xi[i]

        return bias_force

    def get_index(self, xi: np.ndarray) -> list:
        """get list of bin indices for current position of CVs or extended variables

        Args:
            xi (np.ndarray): Current value of collective variable

        Returns:
            bin_x (list):
        """
        bin_x = [-1, -1]
        for i in range(self.ncoords):
            bin_x[i] = int(np.floor(np.abs(xi[i] - self.minx[i]) / self.dx[i]))
        return bin_x

    def unit_conversion_cv(self, *args):
        """convert input to bohr and radians

        args:
            args: arrays to convert of size(dimensions)

        returns:
            args in atomic units
        """
        bohr2angs = 0.52917721092e0  # bohr to angstrom
        deg2rad = np.pi / 180.0  # degree to radians

        for i in range(self.ncoords):
            for arg in args:
                if self.the_cv.type == "angle":
                    arg[i] *= deg2rad
                elif self.the_cv.type == "distance":
                    arg[i] /= bohr2angs
        return args

    def unit_conversion_force(self, *args):
        """convert input to bohr and radians

        args:
            *args: arrays to convert of size(dimensions)

        returns:
            args in atomic units
        """
        bohr2angs = 0.52917721092e0  # bohr to angstrom
        deg2rad = np.pi / 180.0  # degree to radians
        H2KJMOL = 2625.499639

        for i in range(self.ncoords):
            for arg in args:
                if self.cv_type == "angle":
                    arg[i] /= deg2rad / deg2rad / H2KJMOL
                elif self.cv_type == "distance":
                    arg[i] *= bohr2angs * bohr2angs / H2KJMOL
                else:
                    arg[i] /= H2KJMOL
        return args

    def get_cv(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """get state of collective variable

        Returns:
            xi: state of the collective variable
            grad_xi: gradient of the collective variable
        """
        self.the_cv.requires_grad = True
        xi = np.zeros(self.ncoords)
        grad_xi = np.zeros((self.ncoords, len(self.the_md.forces.ravel())))

        for i in range(self.ncoords):
            xi[i], grad_xi[i] = self.the_cv.get_cv(self.cv[i], self.atoms[i], **kwargs)
            self.cv_type[i] = self.the_cv.type

        return xi, grad_xi

    def _kinetics(self, delta_xi):
        """accumulates data for kinetics"""
        forces = self.the_md.get_sampling_data().forces
        m_xi = self._get_mass_of_cv(delta_xi)
        self.mass_traj.append(m_xi)
        self.abs_forces.append(np.linalg.norm(forces))
        self.CV_crit_traj.append(np.dot(delta_xi[0], forces))
        self.abs_grad_xi.append(np.linalg.norm(delta_xi))

    def _get_mass_of_cv(self, delta_xi: np.ndarray) -> np.ndarray:
        """get mass of collective variable for TS theory and kinetics
        only defined for 1D reaction coordinates!

        Args:
            delta_xi: gradients of cv's

        Returns:
            m_xi: coordinate dependent mass of collective variabl
        """
        if self.ncoords == 1:
            return np.dot(delta_xi[0], 1.0 / self.the_md.mass * delta_xi[0])
        else:
            return 0.0

    def write_output(self, data, filename="free_energy.dat"):

        grid = np.copy(self.grid)
        for i in range(self.ncoords):
            if self.the_cv.type == "angle":
                grid *= np.pi / 180.0  # radians to degree
            elif self.the_cv.type == "distance":
                grid /= 0.52917721092e0  # Bohr to Angstrom

        # head of data columns
        out = open(filename, "w")
        for i in range(self.ncoords):
            out.write("%14s\t" % "CV{dim}".format(dim=i))
        for kw in data.keys():
            out.write("%14s\t" % kw)
        out.write("\n")

        # write data to columns
        if self.ncoords == 1:
            for i in range(self.nbins):
                out.write("%14.6f\t" % grid[0][i])
                for dat in data.values():
                    out.write("%14.6f\t" % dat[0][i])
                out.write("\n")

        if self.ncoords == 2:
            for i in range(self.nbins_per_dim[1]):
                for j in range(self.nbins_per_dim[0]):
                    out.write("%14.6f\t%14.6f\t" % (grid[1][i], grid[0][j]))
                    for dat in data.values():
                        out.write("%14.6f\t" % dat[i, j])
                    out.write("\n")
        out.close()

    def _write_traj(self, data: dict = {}):
        """write trajectory of extended or normal ABF at output times

        Args:
            extended: True for use extended system
            additional_data: additional data that one might want to write
        """
        step = self.the_md.get_sampling_data().step

        # convert units to degree and Angstrom
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.traj[:, i] /= np.pi / 180.0
            elif self.cv_type[i] == "distance":
                self.traj[:, i] *= 0.52917721092e0

        # write header
        if not os.path.isfile("CV_traj.dat") and step == 0:
            # start new file in first step

            traj_out = open("CV_traj.dat", "w")
            traj_out.write("%14s\t" % "time [fs]")
            for i in range(len(self.traj[0])):
                traj_out.write("%14s\t" % f"Xi{i}")
            for kw in data.keys():
                traj_out.write("%14s\t" % kw)
            if self.kinetics:
                traj_out.write("%14s\t" % "m_xi [a.u.]")
                traj_out.write("%14s\t" % "|dU| [a.u.]")
                traj_out.write("%14s\t" % "|dxi| [a.u.]")
                traj_out.write("%14s\t" % "dU*dxi [a.u.]")
            traj_out.close()

        elif step > 0:
            # append new steps of trajectory since last output
            traj_out = open("CV_traj.dat", "a")
            for n in range(self.out_freq):
                traj_out.write(
                    "\n%14.6f\t"
                    % (
                        (self.the_md.step - self.out_freq + n)
                        * self.the_md.dt
                        * 1.0327503e0
                    )
                )  # time in fs
                for i in range(len(self.traj[0])):
                    traj_out.write("%14.6f\t" % (self.traj[-self.out_freq + n][i]))
                for val in data.values():
                    traj_out.write("%14.6f\t" % (val[-self.out_freq + n]))
                if self.kinetics:
                    traj_out.write("%14.6f\t" % (self.mass_traj[-self.out_freq + n]))
                    traj_out.write("%14.6f\t" % (self.abs_forces[-self.out_freq + n]))
                    traj_out.write("%14.6f\t" % (self.abs_grad_xi[-self.out_freq + n]))
                    traj_out.write("%14.6f\t" % (self.CV_crit_traj[-self.out_freq + n]))
            traj_out.close()

    def _write_restart(self, filename, **kwargs):
        np.savez(filename, **kwargs)
