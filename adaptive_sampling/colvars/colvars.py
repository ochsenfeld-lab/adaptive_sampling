import sys
import torch
import numpy as np

from typing import Union, Tuple


class CV:
    """Class for Collective Variables

    Automatic computation of gradients is implemented using the autograd differentiation engine of Torch

    Args:
        the_mol: Molecule Object containing masses (natoms) and coords (3*natoms)
        requires_grad: if True, partial derivatives of CVs with respect
                                to atom coordinats are computed
                                and saved to self.gradient
    """

    def __init__(self, the_mol: object, requires_grad: bool = False):

        self.BOHR2ANGS = 0.52917721092e0  # bohr to angstrom conversion factor

        self.the_mol = the_mol
        if not hasattr(the_mol, "masses") or not hasattr(the_mol, "coords"):
            raise ValueError(" >> fatal error: CV missing masses or coords of molecule")

        self.mass = torch.from_numpy(self.the_mol.masses)
        self.coords = torch.from_numpy(self.the_mol.coords.ravel())
        self.natoms = len(self.mass)
        self.requires_grad = requires_grad
        self.gradient = None
        self.cv = None
        self.type = 'default'

    def update_coords(self):
        """The coords tensor and ndarray share the same memory.
        Modifications to the tensor will be reflected in the ndarray and vice versa!"""
        self.coords = torch.from_numpy(self.the_mol.coords.ravel())

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
            for _, a in enumerate(atoms):
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
            for _, a in enumerate(atoms):
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

    def linear_combination(self, cv_def: list) -> Tuple[float, np.ndarray]:
        """linear combination angles or dihedrals between atoms or groups of atoms

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

    def distorted_distance(self, cv_def: list, r_0: float = 3.0) -> float:
        """distorted distance between two mass centers in range(0, inf)

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

        r_0 /= self.BOHR2ANGS

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

    def cec(
        self,
        cv_def: list,
        modified: bool = True,
        r_sw: float = 1.3,
        d_sw: float = 0.05,
        n_pair: int = 15,
    ) -> float:
        """Center of Excess Charge coordinate for long range proton transfer projected on axis of proton transport after Koenig et al.

        Args:
            cv_dev: Definition of proton wire for long range PT,
                [[protons],                  # [all proton indices]
                [donors/acceptors],          # [[weight, index of first donor], [W1, X1], ..., [weight, index of last acceptor]]
                [coupled donors/acceptors]]  # [[weight of group, donor, acceptor], ...]
            modified: if True, use CEC modification by König et al.
            r_sw: switching distance in Angstrom
            d_sw: in Angstrom, controls how fast switching function flips from 0 to 1
            n_pair: exponent for calculation of m(X,{H})

        Returns:
            cv (float): CEC coordinate
        """
        self.update_coords()

        r_sw /= self.BOHR2ANGS
        d_sw /= self.BOHR2ANGS

        self.cv = 0.0

        # vector from first donor to last acceptor defines 1D axis of proton transport
        ind_don, ind_acc = cv_def[1][0][1], cv_def[1][-1][1]
        r_don, _ = self._get_com(ind_don)
        r_acc, _ = self._get_com(ind_acc)

        z = r_acc - r_don
        z_n = 1.0 / torch.linalg.norm(z)
        z_u = z * z_n

        # proton terms
        index_h = cv_def[0]
        coords_h = []
        for _, ind_h in enumerate(index_h):
            (r_hi, _) = self._get_com(ind_h)
            coords_h.append(r_hi)
            r = r_hi - r_don
            self.cv += torch.dot(r, z_u)

        # donor/acceptor terms
        index_x = cv_def[1]
        coords_x = []
        for _, ind_x in enumerate(index_x[1:]):
            (r_xj, _) = self._get_com(ind_x[1])
            coords_x.append(r_xj)
            w_xj = ind_x[0] + 1.0
            r = r_xj - r_don
            self.cv -= w_xj * torch.dot(r, z_u)

        # modified CEC for long range proton transfer
        if modified:
            for _, (ind_h, r_hi) in enumerate(zip(index_h, coords_h)):
                for _, (ind_x, r_xj) in enumerate(zip(index_x, coords_x)):
                    r = r_hi - r_xj
                    f_sw = self._f_sw(r, r_sw, d_sw)
                    self.cv -= f_sw * torch.dot(r, z_u)

        variables = (r_don, r_acc) + tuple(coords_h) + tuple(coords_x)
        indices = [ind_don, ind_acc] + index_h + [ind_x[1] for ind_x in index_x[1:]]

        # correction for coupled donor and acceptor (e.g. for glutamate, aspartate, histidine, ...)
        if len(cv_def) == 3:
            w_pair = [j[0] for j in cv_def[2]]
            ind_pair = [[j[1], j[2]] for j in cv_def[2]]
            coords_pair = []
            for _, (w_pj, ind_pj) in enumerate(zip(w_pair, ind_pair)):

                r_k, _ = self._get_com(ind_pj[0])
                r_l, _ = self._get_com(ind_pj[1])

                coords_pair += r_k
                coords_pair += r_l

                r_kl = torch.dot(r_l - r_k, z_u)
                w = w_pj / 2.0

                # accumulators for m_k and m_l
                denom_k, num_k = 0.0, 0.0
                denom_l, num_l = 0.0, 0.0

                # compute m_k, m_l and their derivatives
                for _, r_hi in enumerate(coords_h):
                    r_ki = r_hi - r_k
                    r_li = r_hi - r_l
                    f_k = self._f_sw(r_ki, r_sw, d_sw)
                    f_l = self._f_sw(r_li, r_sw, d_sw)

                    # for heavy atoms sum over all protons contributes to gradient
                    denom_k += np.power(f_k, n_pair)
                    num_k += np.power(f_k, n_pair + 1)
                    denom_l += np.power(f_l, n_pair)
                    num_l += np.power(f_l, n_pair + 1)

                # add coupled term to xi
                m_k = num_k / denom_k
                m_l = num_l / denom_l
                self.cv += w * (m_k * r_kl - m_l * r_kl)

                indices += [ind_pj[0], ind_pj[1]]
                variables += (r_k, r_l)

        if self.requires_grad:
            atom_grads = self.partial_derivative(self.cv, tuple(variables))
            self.gradient = np.zeros(3 * self.natoms, dtype=float)
            for grad, atom in zip(atom_grads, indices):
                self._grad_to_molecule(grad.numpy(), atom)

        return float(self.cv)

    def gmcec(
        self,
        cv_def: list,
        r_sw: float = 1.3,
        d_sw: float = 0.05,
        n_pair: int = 15,
        mapping: str = "default",
        c: float = 1.0,
    ) -> float:
        """modified CEC coordinate to describe long range proton transfer generalized to complex 3D wire geometries after König et al.

        Args:
            cv_def: Definition of proton wire for long range PT,
                [[protons],                  # [all proton indices]
                [donors/acceptors],          # [[weight, index of first donor], [W1, X1], ..., [weight, index of last acceptor]]
                [coupled donors/acceptors]]  # [[weight of group, donor, acceptor], ...]
            r_sw: switching distance in Angstrom
            d_sw: in Angstrom, controls how fast switching function flips from 0 to 1
            n_pair: exponent for calculation of m(X,{H})
            mapping: 'f_SW': uses switching function (chi = c/(1+e^(d_acc_xi)) - c/(1+e^(d_don_xi)))
                     'fraction': chi = d_don_xi / (d_don_xi+d_acc_xi)
                     'default': antisymmetric stretch between donor, xi and acceptor (chi = d_xi_don - d_xi_acc)

        Returns:
            cv: gmCEC coordinate
        """
        self.update_coords()

        r_sw /= self.BOHR2ANGS
        d_sw /= self.BOHR2ANGS

        # 3D mCEC vector
        xi = torch.zeros(3, dtype=torch.float)

        # protons
        index_h = cv_def[0]
        coords_h = []
        for _, ind_h in enumerate(index_h):
            (r_hi, _) = self._get_com(ind_h)
            coords_h.append(r_hi)
            xi += r_hi

        # donors/acceptors
        w_x = [j[0] + 1.0 for j in cv_def[1]]
        index_x = [j[1] for j in cv_def[1]]
        coords_x = []
        for _, (ind_xj, w_xj) in enumerate(zip(index_x, w_x)):
            (r_xj, _) = self._get_com(ind_xj)
            coords_x.append(r_xj)
            xi -= w_xj * r_xj

        # modified CEC
        for _, r_hi in enumerate(coords_h):
            for _, r_xj in enumerate(coords_x):
                r_ij = r_hi - r_xj
                f_sw = 1.0 / (1.0 + torch.exp((torch.linalg.norm(r_ij) - r_sw) / d_sw))
                xi -= f_sw * r_ij

        variables = tuple(coords_h) + tuple(coords_x)
        indices = index_h + index_x

        # correction for coupled donor and acceptor (e.g. for glutamate, aspartate, histidine, ...)
        if len(cv_def) == 3:
            w_pair = [j[0] for j in cv_def[2]]
            ind_pair = [[j[1], j[2]] for j in cv_def[2]]
            coords_pair = []
            for _, (w_pj, ind_pj) in enumerate(zip(w_pair, ind_pair)):

                r_k, _ = self._get_com(ind_pj[0])
                r_l, _ = self._get_com(ind_pj[1])

                coords_pair += r_k
                coords_pair += r_l

                r_kl = r_l - r_k
                w = w_pj / 2.0

                # accumulators for m_k and m_l
                denom_k, num_k = 0.0, 0.0
                denom_l, num_l = 0.0, 0.0

                # compute m_k, m_l and their derivatives
                for _, r_hi in enumerate(coords_h):
                    r_ki = r_hi - r_k
                    r_li = r_hi - r_l
                    f_k = self._f_sw(r_ki, r_sw, d_sw)
                    f_l = self._f_sw(r_li, r_sw, d_sw)

                    # for heavy atoms sum over all protons contributes to gradient
                    denom_k += torch.pow(f_k, n_pair)
                    num_k += torch.pow(f_k, n_pair + 1)
                    denom_l += torch.pow(f_l, n_pair)
                    num_l += torch.pow(f_l, n_pair + 1)

                # add coupled term to xi
                m_k = num_k / denom_k
                m_l = num_l / denom_l
                xi += w * (m_k * r_kl - m_l * r_kl)

                indices += [ind_pj[0], ind_pj[1]]
                variables += (r_k, r_l)

        # mapping to 1D
        if mapping == "f_SW":
            self.cv = -c / (1.0 + torch.exp(torch.linalg.norm(xi - coords_x[0])))
            self.cv += c / (1.0 + torch.exp(torch.linalg.norm(xi - coords_x[-1])))

        elif mapping == "fraction":
            d_xi_don = torch.linalg.norm(xi - coords_x[0])
            d_xi_acc = torch.linalg.norm(xi - coords_x[-1])
            self.cv = d_xi_don / (d_xi_don + d_xi_acc)

        else:  # default
            self.cv = torch.linalg.norm(xi - coords_x[0]) - torch.linalg.norm(
                xi - coords_x[-1]
            )

        # gradient of gmcec coordinate
        if self.requires_grad:
            atom_grads = self.partial_derivative(self.cv, tuple(variables))
            self.gradient = np.zeros(3 * self.natoms, dtype=float)
            for grad, atom in zip(atom_grads, indices):
                self._grad_to_molecule(grad.numpy(), atom)

        return float(self.cv)

    @staticmethod
    def _f_sw(r: torch.tensor, r_sw: float, d_sw: float) -> float:
        """switching function f_sw(r)"""
        d = torch.linalg.norm(r)
        return 1.0 / (1.0 + torch.exp((d - r_sw) / d_sw))

    def _grad_to_molecule(self, grad_atom: np.ndarray, index_atom: int):
        """atom gradient of size(3) to molecular gradient of size(3N)"""
        index_atom = int(index_atom)
        self.gradient[3 * index_atom] += grad_atom[0]
        self.gradient[3 * index_atom + 1] += grad_atom[1]
        self.gradient[3 * index_atom + 2] += grad_atom[2]

    def custom_lin_comb(self, cvs: list, **kwargs):
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

    def get_cv(self, cv, atoms, **kwargs):
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
        elif cv.lower() == "distorted_distance":
            xi = self.distorted_distance(atoms)
        elif cv.lower() == "cec":
            xi = self.cec(atoms, modified=False, **kwargs)
            self.type = "distance"
        elif cv.lower() == "mcec":
            xi = self.cec(atoms, modified=True, **kwargs)
            self.type = "distance"
        elif cv.lower() == "gmcec":
            xi = self.gmcec(atoms, mapping="stretch", **kwargs)
            self.type = "distance"
        elif cv.lower() == "gmcec1":
            xi = self.gmcec(atoms, mapping="f_SW", **kwargs)
            self.type = None
        elif cv.lower() == "gmcec2":
            xi = self.gmcec(atoms, mapping="fraction", **kwargs)
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


if __name__ == "__main__":

    class MD:
        def __init__(self, mass, coords):
            self.masses = np.array(mass)
            self.coords = np.array(coords)
            self.natoms = len(mass)
            self.forces = np.zeros(3 * self.natoms)

    def four_particles():
        masses = [2, 1, 1, 10]
        coords = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
        return MD(masses, coords)

    the_mol = four_particles()
    cv = CV(the_mol, requires_grad=True)
    f, grad = cv.get_cv("distance", [0, 2])
    print("\nDistance: ", f)
    print("Gradient:\n", grad)

    the_mol.coords = np.array([0, 0, 0, 10, 0, 0, 1, 1, 0, 1, 1, 1])
    f, grad = cv.get_cv("distance", [0, 2])
    print("\nDistance: ", f)
    print("Gradient:\n", grad)

    f = cv.angle([0, 1, 2])
    print("\nAngle: ", np.degrees(f))
    print("Gradient:\n", cv.gradient)

    f = cv.torsion([0, 1, 2, 3])
    print("\nDihedral: ", np.degrees(f))
    print("Gradient:\n", cv.gradient)

    f = cv.linear_combination([[1.0, [2, 3]], [2.0, [0, 1]]])
    print("\nLinear combination: ", f)
    print("Gradient:\n", cv.gradient)

    f = cv.distorted_distance([0, [2, 3]])
    print("\ndistorted distance: ", f)
    print("Gradient:\n", cv.gradient)

    d = cv.cec([[0, 1], [[1.0, 2], [2.0, 3]]])
    print("\ncec: ", d)
    print("Gradient:\n", cv.gradient)

    d = cv.gmcec([[0, 1], [[1.0, 2], [1.0, 3]], [[2, 2, 3]]], mapping="default")
    print("\ngmcec: ", d)
    print("Gradient:\n", cv.gradient)

    f, grad = cv.get_cv(
        "lin_comb_custom", [["distance", 1.0, [0, 2]], ["distance", 1.0, [0, 2]]]
    )
    print("\ncustom linear combination: ", f)
    print("Gradient:\n", cv.gradient)
