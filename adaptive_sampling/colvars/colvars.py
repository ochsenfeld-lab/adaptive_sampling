import sys
import torch
import numpy as np

from typing import Union, Tuple
from ..interface.sampling_data import MDInterface
from ..units import *


class CV:
    """Class for Collective Variables

    Automatic computation of gradients is implemented using the autograd differentiation engine of Torch

    Args:
        the_mol: Molecule Object containing masses (natoms) and coords (3*natoms)
        requires_grad: if True, partial derivatives of CVs with respect
                                to atom coordinats are computed
                                and saved to self.gradient
    """

    def __init__(self, the_mol: MDInterface, requires_grad: bool = False):

        self.the_mol = the_mol

        md_state = self.the_mol.get_sampling_data()

        self.mass = torch.from_numpy(md_state.mass)
        self.coords = torch.from_numpy(md_state.coords.ravel())
        self.natoms = len(self.mass)
        self.requires_grad = requires_grad
        self.gradient = None
        self.cv = None
        self.type = "default"

    def update_coords(self):
        """The coords tensor and ndarray share the same memory.
        Modifications to the tensor will be reflected in the ndarray and vice versa!"""
        self.coords = torch.from_numpy(self.the_mol.get_sampling_data().coords.ravel())

    @staticmethod
    def partial_derivative(
        f: torch.tensor, *args: Tuple[torch.tensor]
    ) -> Tuple[torch.tensor]:
        """get partial derivative of arbitrary function from torch.autograd

        Args:
            f (torch.tensor): function f(*args) to differentiate
            *args (torch.tensor): variables of f for which derivative is computed

        Returns:
            partial_derivatives (Tuple[torch.tensor]): derivatives of f with respect to args
        """
        return torch.autograd.grad(f, *args)

    def _get_com(self, atoms: Union[int, list]) -> Tuple[torch.tensor, float]:
        """get center of mass (com) of group of atoms
        if self.require_grad = True partial derivative with respect to com can be calculated using torch.autograd.grad

        Args:
            atoms (Union[int, list]): atom index or list of atom indices

        Returns:
            com (torch.tensor): Center of Mass
            m_tot(float): Total mass of involved atoms
        """
        if hasattr(atoms, "__len__"):
            # compute center of mass for group of atoms
            com = torch.zeros(3, dtype=torch.float)
            for a in atoms:
                a = int(a)
                com += self.coords[3 * a] * self.mass[a]
                com += self.coords[3 * a + 1] * self.mass[a]
                com += self.coords[3 * a + 2] * self.mass[a]

            m_tot = self.mass[atoms].sum()
            com /= m_tot

        else:
            # only one atom
            atom = int(atoms)
            m_tot = self.mass[atom]
            com = torch.tensor(
                [
                    self.coords[3 * atom],
                    self.coords[3 * atom + 1],
                    self.coords[3 * atom + 2],
                ]
            )

        com = com.float()
        com.requires_grad = self.requires_grad

        return com, m_tot

    def _get_atom_weights(
        self, mass_group: float, atoms: Union[int, list]
    ) -> torch.tensor:
        """get mass weights of atoms for gradient of group of atoms

        Args:
            mass_group (float): sum of mass of atoms
            atoms (Union[int, list]): atom index or list of atom indices

        Returns:
            coords (torch.tensor): 3*3N array of mass weights of atoms
        """
        coords = torch.zeros((3, 3 * self.natoms))
        if hasattr(atoms, "__len__"):
            for a in atoms:
                a = int(a)
                coords[0, 3 * a] = self.mass[a] / mass_group
                coords[1, 3 * a + 1] = self.mass[a] / mass_group
                coords[2, 3 * a + 2] = self.mass[a] / mass_group
        else:
            atoms = int(atoms)
            coords[0, 3 * atoms] = 1.0
            coords[1, 3 * atoms + 1] = 1.0
            coords[2, 3 * atoms + 2] = 1.0

        return coords

    def x(self) -> float:
        """use x axis as cv for numerical examples"""
        self.update_coords()
        if self.requires_grad:
            self.gradient = np.array([1.0, 0.0])
        self.cv = self.coords[0]
        return self.cv

    def y(self) -> float:
        """use y axis as cv for numerical examples"""
        self.update_coords()
        if self.requires_grad:
            self.gradient = np.array([0.0, 1.0])
        self.cv = self.coords[1]
        return self.cv

    def distance(self, cv_def: list) -> float:
        """distance between two mass centers in range(0, inf)

        Args:
            cv_def (list):
                distance beteen atoms: [ind0, ind1]
                distance between mass centers: [[ind00, ind01, ...], [ind10, ind11, ...]]

        Returns:
            cv (float): computed distance
        """
        if len(cv_def) != 2:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of distance!"
            )

        self.update_coords()

        (p1, m0) = self._get_com(cv_def[0])
        (p2, m1) = self._get_com(cv_def[1])

        # get distance
        r12 = p2 - p1
        self.cv = torch.linalg.norm(r12, dtype=torch.float)

        # get forces
        if self.requires_grad:
            # @AH: why don't you do? Does this not work?
            #            self.gradient = self.partial_derivative(self.cv, self.coords)
            atom_grads = self.partial_derivative(self.cv, (p1, p2))
            self.gradient = torch.matmul(
                atom_grads[0], self._get_atom_weights(m0, cv_def[0])
            )
            self.gradient += torch.matmul(
                atom_grads[1], self._get_atom_weights(m1, cv_def[1])
            )
            self.gradient = self.gradient.numpy()

        return float(self.cv)

    def angle(self, cv_def: list) -> float:
        """get angle between three mass centers in range(-pi,pi)

        Args:
            cv_def (list):
                angle between two atoms: [ind0, ind1, ind3]
                angle between centers of mass: [[ind00, ind01, ...], [ind10, ind11, ...], [ind20, ind21, ...]]

        Returns:
            cv (float): computed angle
        """
        if len(cv_def) != 3:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of angle!"
            )

        self.update_coords()

        (p1, m0) = self._get_com(cv_def[0])
        (p2, m1) = self._get_com(cv_def[1])
        (p3, m2) = self._get_com(cv_def[2])

        # get angle
        q12 = p1 - p2
        q23 = p2 - p3

        q12_n = torch.linalg.norm(q12)
        q23_n = torch.linalg.norm(q23)

        q12_u = q12 / q12_n
        q23_u = q23 / q23_n

        self.cv = torch.arccos(torch.dot(-q12_u, q23_u))

        # get forces
        if self.requires_grad:
            atom_grads = self.partial_derivative(self.cv, (p1, p2, p3))
            self.gradient = torch.matmul(
                atom_grads[0], self._get_atom_weights(m0, cv_def[0])
            )
            self.gradient += torch.matmul(
                atom_grads[1], self._get_atom_weights(m1, cv_def[1])
            )
            self.gradient += torch.matmul(
                atom_grads[2], self._get_atom_weights(m2, cv_def[2])
            )
            self.gradient = self.gradient.numpy()

        return float(self.cv)

    def torsion(self, cv_def: list) -> float:
        """torsion angle between four mass centers in range(-pi,pi)

        Args:
            cv_def (list):
                dihedral between atoms: [ind0, ind1, ind2, ind3]
                dihedral between center of mass: [[ind00, ind01, ...],
                                                  [ind10, ind11, ...],
                                                  [ind20, ind21, ...],
                                                  [ind30, ind 31, ...]]

        Returns:
            cv (float): computed torsional angle
        """
        if len(cv_def) != 4:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of dihedral!"
            )

        self.update_coords()

        (p1, m1) = self._get_com(cv_def[0])
        (p2, m2) = self._get_com(cv_def[1])
        (p3, m3) = self._get_com(cv_def[2])
        (p4, m4) = self._get_com(cv_def[3])

        # get dihedral
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3

        q23_u = q23 / torch.linalg.norm(q23)

        n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
        n2 = q34 - torch.dot(q34, q23_u) * q23_u

        self.cv = torch.atan2(torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2))

        # get forces
        if self.requires_grad:
            atom_grads = self.partial_derivative(self.cv, (p1, p2, p3, p4))
            self.gradient = torch.matmul(
                atom_grads[0], self._get_atom_weights(m1, cv_def[0])
            )
            self.gradient += torch.matmul(
                atom_grads[1], self._get_atom_weights(m2, cv_def[1])
            )
            self.gradient += torch.matmul(
                atom_grads[2], self._get_atom_weights(m3, cv_def[2])
            )
            self.gradient += torch.matmul(
                atom_grads[3], self._get_atom_weights(m4, cv_def[3])
            )
            self.gradient = self.gradient.numpy()

        return float(self.cv)

    def linear_combination(self, cv_def: list) -> float:
        """linear combination of distances, angles or dihedrals between atoms or groups of atoms

        Args:
            cv_dev (list):
                list of distances, angle or torsions with prefactors: [[fac0, [ind00, ind01]],
                                                                       [fac1, [ind10, ind11, ind12]],
                                                                       [fac2, [ind20, ind21, ind22, ind23]],
                                                                       ...]

        Returns:
            cv (float): linear combination of distances/angles/dihedrals
        """
        self.update_coords()

        self.lc_contribs = []
        gradient = np.zeros(3 * self.natoms, dtype=float)

        for cv in cv_def:

            if len(cv[1]) == 2:
                x = self.distance(cv[1])

            elif len(cv[1]) == 3:
                x = self.angle(cv[1])

            elif len(cv[1]) == 4:
                x = self.torsion(cv[1])

            else:
                raise ValueError(
                    "CV ERROR: Invalid number of centers in definition of linear combination!"
                )

            self.lc_contribs.append(cv[0] * x)
            if self.requires_grad:
                gradient += cv[0] * self.gradient

        if self.requires_grad:
            self.gradient = gradient

        self.cv = np.asarray(self.lc_contribs).sum()
        return float(self.cv)

    def write_lc_traj(self, out: str = "lc_traj.dat"):
        """write out seperate trajectory for contributions to linear combination

        Args:
            our (str): name of output file
        """
        with open(out, "a") as traj_out:
            for lc in self.lc_contribs:
                traj_out.write("%14.6f\t" % lc)
            traj_out.write("\n")

    def coordination_number(self, cv_def: list, r_0: float = 3.0) -> float:
        """coordination number between two mass centers in range(0, inf) mapped to range(1,0)

        Args:
            cv_def (list):
                distorted distance beteen atoms: [ind0, ind1]
                distorted distance between mass centers: [[ind00, ind01, ...],
                                                          [ind10, ind11, ...]]

        Returns:
            distorted distance (float): computed distance
        """
        self.update_coords()

        if len(cv_def) != 2:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of distance!"
            )

        r_0 /= BOHR_to_ANGSTROM

        (p1, m0) = self._get_com(cv_def[0])
        (p2, m1) = self._get_com(cv_def[1])

        # get distance
        r12 = p2 - p1
        d = torch.linalg.norm(r12, dtype=torch.float) / r_0

        self.cv = (1.0 - torch.pow(d, 6)) / (1.0 - torch.pow(d, 12))

        # get forces
        if self.requires_grad:
            atom_grads = self.partial_derivative(self.cv, (p1, p2))
            self.gradient = torch.matmul(
                atom_grads[0], self._get_atom_weights(m0, cv_def[0])
            )
            self.gradient += torch.matmul(
                atom_grads[1], self._get_atom_weights(m1, cv_def[1])
            )
            self.gradient = self.gradient.numpy()

        return float(self.cv)

    def custom_lin_comb(self, cvs: list, **kwargs) -> float:
        """custom linear combination of arbitrary functions"""
        self.update_coords()
        cv = 0.0
        gradient = np.zeros(len(self.gradient))
        for _, cv_def in enumerate(cvs):
            z, dz = self.get_cv(cv_def[0], cv_def[2])
            cv += cv_def[1] * z
            gradient += cv_def[1] * dz

        self.cv = cv
        self.gradient = gradient
        return float(cv)

    def get_cv(self, cv, atoms, **kwargs) -> Tuple[float, np.ndarray]:
        """get state of collective variable

        Returns:
           xi (float): value of collective variable
           gradient (np.ndarray) : gradient of collective variable
        """
        if cv.lower() == "x":
            xi = self.x()
            self.type = "2d"
        elif cv.lower() == "y":
            xi = self.y()
            self.type = "2d"
        elif cv.lower() == "distance":
            xi = self.distance(atoms)
            self.type = "distance"
        elif cv.lower() == "angle":
            xi = self.angle(atoms)
            self.type = "angle"
        elif cv.lower() == "torsion":
            xi = self.torsion(atoms)
            self.type = "angle"
        elif cv.lower() == "lin_comb_dists":
            xi = self.linear_combination(atoms)
            self.type = "distance"
        elif cv.lower() == "lin_comb_angles":
            xi = self.linear_combination(atoms)
            self.type = "angle"
        elif cv.lower() == "linear_combination":
            xi = self.linear_combination(atoms)
            self.type = None
        elif cv.lower() == "coordination_number":
            xi = self.coordination_number(atoms)
            self.type = None
        elif cv.lower() == "lin_comb_custom":
            xi = self.custom_lin_comb(atoms)
            self.type = None
        else:
            print(" >>> Error in CV: Unknown coordinate")
            sys.exit(1)

        if self.requires_grad:
            return xi, self.gradient
        else:
            return xi
