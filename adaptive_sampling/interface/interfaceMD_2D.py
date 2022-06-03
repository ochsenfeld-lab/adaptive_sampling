#!/usr/bin/env python
import sys
import random
import numpy as np
from typing import Tuple
from .sampling_data import SamplingData
from ..units import *

class MD:
    """Class for MD on test potentials"""

    def __init__(
        self,
        mass_in=1,
        coords_in=[0.0, 0.0],
        potential="1",
        dt_in=0.1e0,
        target_temp_in=298.15e0,
        seed_in=4911,
    ):
        # Init Vars

        self.step = 0
        self.coords = np.array(coords_in)
        self.natoms = 1
        self.dt_fs = dt_in
        self.dt = dt_in / atomic_to_fs
        self.target_temp = target_temp_in
        self.forces = np.zeros(2 * self.natoms)
        self.momenta = np.zeros(2 * self.natoms)
        self.potential = potential

        # Random Number Generator
        if type(seed_in) is int:
            random.seed(seed_in)
            print("THE RANDOM NUMBER SEED WAS: %i" % (seed_in))
        else:
            try:
                random.setstate(seed_in)
            except:
                raise ValueError(
                    "\tThe provided seed was neither an int nor a state of random"
                )

        # Mass
        self.mass = mass_in
        self.conf_forces = np.zeros(2 * self.natoms)
        self.masses = np.zeros(2 * self.natoms)
        self.masses[0] = self.mass
        self.masses[1] = self.mass

        # Ekin and print
        self.epot = 0.0e0
        self.ekin = 0.0e0
        self.temp = 0.0e0
        self.vol = 0.0e0

    # -----------------------------------------------------------------------------------------------------
    def calc_init(self, init_temp=298.15e0):
        """Initial calculation of energy, forces, momenta

        Args:
           init_momenta_in         (string,random,zero/random/read),
           momena_in               (list,np.array([]))
           init_temp               (float, 298.15)

        Returns:
           -
        """
        (self.epot, self.forces) = self.calc_energy_forces_MD(self.potential)

        # Init momenta random
        self.momenta = np.zeros(2 * self.natoms)
        self.momenta[0] = random.gauss(0.0, 1.0) * np.sqrt(init_temp * self.mass)
        self.momenta[1] = random.gauss(0.0, 1.0) * np.sqrt(init_temp * self.mass)

        TTT = (np.power(self.momenta, 2) / self.masses).sum() / 2.0
        self.momenta *= np.sqrt(init_temp / (TTT * atomic_to_K))

    # -----------------------------------------------------------------------------------------------------
    def calc(self) -> Tuple[float, np.ndarray]:
        """Calculation of energy, forces

        Returns:
           energy (float): energy,
           forces (ndarray): forces
        """
        (self.epot, self.forces) = self.calc_energy_forces_MD(self.potential)

    # -----------------------------------------------------------------------------------------------------
    def calc_energy_forces_MD(self, potential: str = "1") -> tuple:
        """Calculate energy and forces

        Args:
            potential: selects potential energy function
        """
        x = self.coords[0]
        y = self.coords[1]
        self.forces = np.zeros(2 * self.natoms)

        d = 40.0
        e = 20.0

        if potential == "1":

            a = 8.0e-6 / atomic_to_kJmol
            b = 0.5 / atomic_to_kJmol
            d = 80.0
            e = 160.0

            s1 = (x - d) * (x - d)
            s2 = (x - e) * (x - e)

            self.epot = a * s1 * s2 + b * y * y

            self.forces[0] = 2.0 * a * ((x - d) * s2 + s1 * (x - e))
            self.forces[1] = 2.0 * b * y

        elif potential == "2":

            a = 0.005
            b = 0.040

            exp_1 = np.exp((-a * (x - d) * (x - d)) + (-b * (y - e) * (y - e)))
            exp_2 = np.exp((-a * (x + d) * (x + d)) + (-b * (y + e) * (y + e)))

            self.epot = -np.log(exp_1 + exp_2) / atomic_to_kJmol

            self.forces[0] = (
                -(
                    (-2.0 * a * (x - d) * exp_1 - 2.0 * a * (x + d) * exp_2)
                    / (exp_1 + exp_2)
                )
                / atomic_to_kJmol
            )
            self.forces[1] = (
                -(
                    (-2.0 * b * (y - e) * exp_1 - 2.0 * b * (y + e) * exp_2)
                    / (exp_1 + exp_2)
                )
                / atomic_to_kJmol
            )

        else:
            print("\n\tInvalid Potential!")
            sys.exit(1)

        return (self.epot, self.forces)

    # -----------------------------------------------------------------------------------------------------
    def calc_etvp(self):
        """Calculation of kinetic energy, total energy, volume, and pressure

        Args:
           -

        Returns:
           -
        """
        self.ekin = (np.power(self.momenta, 2) / self.masses).sum()
        self.ekin /= 2.0

        self.temp = self.ekin / kB_in_atomic

    # -----------------------------------------------------------------------------------------------------
    def propagate(self, langevin=True, friction=1.0e-3):
        """Propagate momenta/coords with Velocity Verlet

        Args:
           langevin                (bool, True)
           friction                (float, 10^-3 1/fs)
        """
        if langevin == True:
            prefac = 2.0 / (2.0 + friction * self.dt_fs)
            rand_push = np.sqrt(self.target_temp * friction * self.dt_fs * kB_in_atomic / 2.0e0)
            self.rand_gauss = np.zeros(shape=(len(self.momenta),), dtype=np.double)
            self.rand_gauss[0] = random.gauss(0, 1)
            self.rand_gauss[1] = random.gauss(0, 1)

            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt * self.forces
            self.coords += prefac * self.dt * self.momenta / self.masses

        else:
            self.momenta -= 0.5e0 * self.dt * self.forces
            self.coords += self.dt * self.momenta / self.masses

    # -----------------------------------------------------------------------------------------------------
    def up_momenta(self, langevin=True, friction=1.0e-3):
        """Update momenta with Velocity Verlet

        Args:
           langevin                (bool, True)
           friction                (float, 10^-3 1/fs)           
        """
        if langevin == True:
            prefac = (2.0e0 - friction * self.dt_fs) / (2.0e0 + friction * self.dt_fs)
            rand_push = np.sqrt(self.target_temp * friction * self.dt_fs * kB_in_atomic / 2.0e0)
            self.momenta *= prefac
            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt * self.forces
        else:
            self.momenta -= 0.5e0 * self.dt * self.forces

    # -----------------------------------------------------------------------------------------------------
    def get_sampling_data(self):
        """interface to adaptive_sampling"""
        return SamplingData(
            self.mass,    
            self.coords,
            self.forces,
            self.epot,
            self.temp,
            self.natoms,
            self.step,
            self.dt_fs,
        )
