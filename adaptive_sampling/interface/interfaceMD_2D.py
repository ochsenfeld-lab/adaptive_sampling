#!/usr/bin/env python
import random
import numpy as np
from typing import Tuple
from .sampling_data import SamplingData
from adaptive_sampling import units


class MD:
    """Class for MD on test potentials"""

    def __init__(
        self,
        mass_in=1.0,
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
        self.dt = dt_in / units.atomic_to_fs_times_sqrt_amu2au
        self.target_temp = target_temp_in
        self.forces = np.zeros(2 * self.natoms)
        self.qm_forces = np.zeros(2 * self.natoms)
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
        self.conf_traj = []
        self.masses = np.full(2, self.mass)

        # Ekin and print
        self.epot = 0.0e0
        self.qm_energy = 0.0e0
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
        self.momenta *= np.sqrt(init_temp / (TTT * units.atomic_to_K))

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
        import torch

        coords = torch.from_numpy(self.coords)
        coords.requires_grad = True

        x = coords[0]
        y = coords[1]

        d = 40.0
        e = 20.0

        if potential == "1":

            a = 8.0e-6 / units.atomic_to_kJmol
            b = 0.5 / units.atomic_to_kJmol
            d = 80.0
            e = 160.0

            s1 = (x - d) * (x - d)
            s2 = (x - e) * (x - e)

            self.epot = a * s1 * s2 + b * y * y

        elif potential == "2":

            a = 0.005
            b = 0.040

            exp_1 = torch.exp((-a * (x - d) * (x - d)) + (-b * (y - e) * (y - e)))
            exp_2 = torch.exp((-a * (x + d) * (x + d)) + (-b * (y + e) * (y + e)))

            self.epot = -torch.log(exp_1 + exp_2) / units.atomic_to_kJmol

        elif potential == "3":

            B = 1.0 / units.atomic_to_kJmol
            A = B * torch.tensor([-40.0, -10.0, -34.0, 3.0])
            alpha = torch.tensor([-1.00, -1.00, -6.50, 0.7])
            beta = torch.tensor([0.00, 0.00, 11.00, 0.6])
            gamma = torch.tensor([-10.0, -10.0, -6.50, 0.7])
            x0 = torch.tensor([1.0, 0.0, -0.5, -1.0])
            y0 = torch.tensor([0.0, 0.5, 1.5, 1.0])

            self.epot = (
                A
                * torch.exp(
                    alpha * (x - x0) * (x - x0)
                    + beta * (x - x0) * (y - y0)
                    + gamma * (y - y0) * (y - y0)
                )
            ).sum()

        elif potential == "4":

            a = 10e-2
            b = 10.3333e-2
            c = 2.52e-2
            d = 2.0e-2

            s1 = a * torch.square(x)
            s2 = b * torch.pow(x, 3)
            s3 = c * torch.pow(x, 4)
            s4 = d * torch.square(y)

            self.epot = (
                s1 - s2 + s3 + s4 + 0.02596466400007491
            )  # minimum shifted to zero
        
        elif potential == "5":
            prefac = 0.003
            self.qm_energy = prefac * ( - torch.cos(x * np.pi) + 0.1 * torch.square(x))
            mm_energy = - prefac * torch.cos(y * np.pi)
            self.epot = self.qm_energy + mm_energy

            self.qm_forces = torch.autograd.grad(self.qm_energy, coords, allow_unused=True, retain_graph=True)[0]
            self.qm_forces = self.qm_forces.detach().numpy()
            self.qm_energy = float(self.qm_energy)
        
        elif potential == "6":
            prefac = 0.003
            self.qm_energy = prefac * ( - torch.cos(x * np.pi) + 0.1 * torch.square(x))
            mm_energy =  prefac * ( - torch.cos(y * np.pi) + 0.1 * torch.square(y))
            self.epot = self.qm_energy + mm_energy

            self.qm_forces = torch.autograd.grad(self.qm_energy, coords, allow_unused=True, retain_graph=True)[0]
            self.qm_forces = self.qm_forces.detach().numpy()
            self.qm_energy = float(self.qm_energy)
        
        elif potential == "7":
            prefac = 0.004
            self.qm_energy = prefac * ( - torch.cos(x * np.pi) + 0.1 * torch.square(x))
            mm_energy =  prefac * ( - torch.cos(y * np.pi) + 0.1 * torch.square(y))
            self.epot = self.qm_energy + mm_energy

            self.qm_forces = torch.autograd.grad(self.qm_energy, coords, allow_unused=True, retain_graph=True)[0]
            self.qm_forces = self.qm_forces.detach().numpy()
            self.qm_energy = float(self.qm_energy)

        else:
            return (0.0, np.zeros(2))
            # raise ValueError(" >>> Invalid Potential!")

        self.forces = torch.autograd.grad(self.epot, coords, allow_unused=True)[0]
        self.forces = self.forces.detach().numpy()
        

        return (float(self.epot), self.forces)

    # -----------------------------------------------------------------------------------------------------
    def calc_etvp(self):
        """Calculation of kinetic energy, total energy, volume, and pressure

        Args:
           -

        Returns:
           -
        """
        self.ekin = (np.square(self.momenta) / self.masses).sum()
        self.ekin /= 2.0
        self.temp = (self.ekin * 2.0) / units.kB_in_atomic

    # -----------------------------------------------------------------------------------------------------
    def propagate(self, langevin=True, friction=1.0e-3):
        """Propagate momenta/coords with Velocity Verlet

        Args:
           langevin                (bool, True)
           friction                (float, 10^-3 1/fs)
        """
        if langevin == True:
            prefac = 2.0 / (2.0 + friction * self.dt_fs)
            rand_push = np.sqrt(
                self.target_temp * friction * self.dt_fs * units.kB_in_atomic / 2.0e0
            )
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
            rand_push = np.sqrt(
                self.target_temp * friction * self.dt_fs * units.kB_in_atomic / 2.0e0
            )
            self.momenta *= prefac
            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt * self.forces
        else:
            self.momenta -= 0.5e0 * self.dt * self.forces

    # -----------------------------------------------------------------------------------------------------
    def confine_colvar(self, ats):
        """Confine atom to collective variable (CV)

        Args:
            ats (list): list of lists with definition of confinement
                [cv, grad_cv, r_0, k in kJ/mol]]

        Returns:
            -
        """
        self.conf_forces = np.zeros_like(self.forces)
        for cv in ats:
            d = cv[0] - cv[2]
            k = cv[3] / units.atomic_to_kJmol
            conf_energy = 0.5 * k * d * d
            self.epot += conf_energy
            self.conf_forces += k * d * cv[1]
        self.forces += self.conf_forces
        self.conf_traj.append([cv[0], conf_energy])

    # -----------------------------------------------------------------------------------------------------
    def print_confine(self, name):
        """Saves the current value of the confinements and their energy contributions to a file

        Args:
            name (str): name of output file

        Returns:
            -
        """
        if self.step == 0:
            f = open(name, "w")
        else:
            f = open(name, "a")
        string = str("%20.10e  " % (self.step * self.dt_fs))
        for conf in self.conf_traj[-1]:
            string += str("%20.10e  " % (conf))
        string += "\n"
        f.write(string)
        f.close()

    # -----------------------------------------------------------------------------------------------------
    def get_sampling_data(self):
        """interface to adaptive_sampling"""

        if self.potential in ["5", "6", "7"]:
            return SamplingData(
                self.masses,
                self.coords,
                self.forces,
                self.epot,
                self.temp,
                self.natoms,
                self.step,
                self.dt,
                self.qm_forces,
                self.qm_energy,
            )
        
        return SamplingData(
            self.masses,
            self.coords,
            self.forces,
            self.epot,
            self.temp,
            self.natoms,
            self.step,
            self.dt_fs,
        )
