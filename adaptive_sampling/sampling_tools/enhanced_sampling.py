#!/usr/bin/env python
import sys, os
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from ..interface.sampling_data import MDInterface
from ..colvars.colvars import CV
from .utils import diff
from ..units import *


class EnhancedSampling(ABC):
    """Abstract class for molecular dynamics based sampling algorithms

    Args:
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
            [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
        multiple_walker: share bias with other simulations via buffer file
        periodic: if True, no harmonic walls are applied at boundary of CV 
    """

    def __init__(
        self,
        md: MDInterface,
        cv_def: list,
        equil_temp: float = 300.0,
        verbose: bool = True,
        kinetics: bool = False,
        f_conf: float = 100,
        output_freq: int = 100,
        multiple_walker: bool = False,
        periodic: bool = False,
        **kwargs,
    ):

        self.the_md = md
        self.the_cv = CV(self.the_md, requires_grad=True)
        self.out_freq = output_freq
        self.equil_temp = equil_temp
        self.verbose = verbose
        self.shared = multiple_walker
        self.periodic = periodic

        # definition of CVs
        self.ncoords = len(cv_def)
        self.cv = np.array([item[0] for item in cv_def], dtype=str)
        self.atoms = [cv_def[i][1] for i in range(self.ncoords)]
        self.minx = np.array([item[2] for item in cv_def], dtype=float)
        self.maxx = np.array([item[3] for item in cv_def], dtype=float)
        self.dx = np.array([item[4] for item in cv_def], dtype=float)
        self.f_conf = np.array([f_conf for _ in range(self.ncoords)], dtype=float)

        self.cv_type = ["" for _ in range(self.ncoords)]
        (xi, delta_xi) = self.get_cv(**kwargs)

        # unit conversion
        self.minx, self.maxx, self.dx = self.unit_conversion_cv(
            self.minx, self.maxx, self.dx
        )
        self.f_conf = self.unit_conversion_force(self.f_conf)

        # store trajectories of CVs and temperature and epot between outputs
        md_state = self.the_md.get_sampling_data()
        self.traj = np.array([xi])
        self.temp = [md_state.temp]
        self.epot = [md_state.epot]

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
        if self.kinetics and self.ncoords == 1:
            self.mass_traj = [self._get_mass_of_cv(delta_xi)]
            self.abs_forces = [np.linalg.norm(md_state.forces)]
            self.CV_crit_traj = [np.abs(np.dot(md_state.forces, delta_xi[0]))]
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
                        self.minx[i] * DEGREES_per_RADIAN,
                        self.maxx[i] * DEGREES_per_RADIAN,
                        self.dx[i] * DEGREES_per_RADIAN,
                        "Degree",
                    )
                elif self.cv_type[i] == "distance":
                    output_dat = (
                        self.minx[i] * BOHR_to_ANGSTROM,
                        self.maxx[i] * BOHR_to_ANGSTROM,
                        self.dx[i] * BOHR_to_ANGSTROM,
                        "Angstrom",
                    )
                else:
                    output_dat = (self.minx[i], self.maxx[i], self.dx[i], "")
                print(f"\t Minimum{i}:\t{output_dat[0]} {output_dat[3]}")
                print(f"\t Maximum{i}:\t{output_dat[1]} {output_dat[3]}")
                print(f"\t Bin width{i}:\t{output_dat[2]} {output_dat[3]}")
            print("\t----------------------------------------------")
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

    @abstractmethod
    def write_restart(self):
        pass

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def write_traj(self):
        pass

    def harmonic_walls(
        self,
        xi: np.ndarray,
        delta_xi: np.ndarray,
        margin: np.ndarray = np.array([0, 0]),
    ) -> np.ndarray:
        """confine system with harmonic walls to range(self.minx, self.maxx)

        Args:
            xi: collective variable
            delta_xi: gradient of collective variable
            margin: inset for start of harmonic wall

        Returns:
            bias_force: confinement force
        """
        conf_force = np.zeros_like(self.the_md.get_sampling_data().forces.ravel())
        if self.periodic:
            return conf_force

        for i in range(self.ncoords):
            if xi[i] > (self.maxx[i] - margin[i]):
                r = diff(self.maxx[i] - margin[i], xi[i], self.cv_type[i])
                conf_force -= self.f_conf[i] * r * delta_xi[i]

            elif xi[i] < (self.minx[i] + margin[i]):
                r = diff(self.minx[i] + margin[i], xi[i], self.cv_type[i])
                conf_force -= self.f_conf[i] * r * delta_xi[i]

        return conf_force

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

    def _check_boundaries(self, x):
        """returns True if x is between minx and maxx"""
        return (x < self.maxx).all() and (x > self.minx).all()

    def unit_conversion_cv(self, *args):
        """convert input to atomic units

        Args:
            args: arrays to convert of size(dimensions)

        Returns:
            args in atomic units
        """
        for i in range(self.ncoords):
            for arg in args:
                if self.the_cv.type == "angle":
                    arg[i] /= DEGREES_per_RADIAN
                elif self.the_cv.type == "distance":
                    arg[i] /= BOHR_to_ANGSTROM
        return args

    def unit_conversion_force(self, *args):
        """convert input to atomic units

        Args:
            *args: arrays to convert of size(dimensions)

        Returns:
            args in atomic units
        """

        for i in range(self.ncoords):
            for arg in args:
                if self.cv_type == "angle":
                    arg[i] *= DEGREES_per_RADIAN * DEGREES_per_RADIAN / atomic_to_kJmol
                elif self.cv_type == "distance":
                    arg[i] *= BOHR_to_ANGSTROM * BOHR_to_ANGSTROM / atomic_to_kJmol
                else:
                    arg[i] /= atomic_to_kJmol
        return args

    def get_cv(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """get state of collective variable

        Returns:
            xi: state of the collective variable
            grad_xi: gradient of the collective variable
        """
        self.the_cv.requires_grad = True
        xi = np.zeros(self.ncoords)
        grad_xi = np.zeros(
            (self.ncoords, len(self.the_md.get_sampling_data().forces.ravel()))
        )

        for i in range(self.ncoords):
            xi[i], grad_xi[i] = self.the_cv.get_cv(self.cv[i], self.atoms[i], **kwargs)
            self.cv_type[i] = self.the_cv.type

        return xi, grad_xi

    def _kinetics(self, delta_xi):
        """accumulates data for kinetics"""
        forces = self.the_md.get_sampling_data().forces
        self.mass_traj.append(self._get_mass_of_cv(delta_xi))
        self.abs_forces.append(np.linalg.norm(forces))
        self.CV_crit_traj.append(np.dot(delta_xi[0], forces))
        self.abs_grad_xi.append(np.linalg.norm(delta_xi))

    def _get_mass_of_cv(self, delta_xi: np.ndarray) -> float:
        """get mass of collective variable for TS theory and kinetics
        only defined for 1D reaction coordinates!

        Args:
            delta_xi: gradients of cv's

        Returns:
            m_xi_inv: coordinate dependent mass of collective variabl
        """
        if self.ncoords == 1:
            if self.cv_type[0] == "2d":
                return np.dot(delta_xi[0], (1.0 / np.repeat(self.the_md.mass, 2)) * delta_xi[0])
            return np.dot(delta_xi[0], (1.0 / np.repeat(self.the_md.mass, 3)) * delta_xi[0])

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

    def _write_traj(self, data: dict = {}, filename: str = 'CV_traj.dat'):
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
                            traj_out.write("%14.6f\t" % (self.traj[-self.out_freq + n][i] * DEGREES_per_RADIAN))
                        elif self.cv_type[i] == "distance":
                            traj_out.write("%14.6f\t" % (self.traj[-self.out_freq + n][i] * BOHR_to_ANGSTROM))
                        else:
                            traj_out.write("%14.6f\t" % (self.traj[-self.out_freq + n][i]))

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
