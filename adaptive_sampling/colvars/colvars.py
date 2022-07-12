import sys
import torch
import numpy as np

from typing import Union, Tuple
from ..interface.sampling_data import MDInterface
from ..units import *
from .utils import *
from .kearsley import Kearsley


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
        self.requires_grad = requires_grad

        md_state = self.the_mol.get_sampling_data()

        self.mass = torch.from_numpy(md_state.mass).float()
        self.coords = torch.from_numpy(md_state.coords.ravel()).float()
        self.natoms = len(self.mass)
        self.requires_grad = requires_grad
        self.gradient = None
        self.cv = None
        self.type = "default"
        self.reference = None
        self.reference_internal = None

    def update_coords(self):
        """The coords tensor and ndarray share the same memory.
        Modifications to the tensor will be reflected in the ndarray and vice versa!"""
        self.coords = torch.from_numpy(self.the_mol.get_sampling_data().coords.ravel())
        self.coords = self.coords.float()
        self.coords.requires_grad = self.requires_grad

    def _get_com(self, atoms: Union[int, list]) -> Tuple[torch.tensor, float]:
        """get center of mass (com) of group of atoms

        Args:
            atoms (Union[int, list]): atom index or list of atom indices

        Returns:
            com (torch.tensor): Center of Mass
        """
        if hasattr(atoms, "__len__"):
            # compute center of mass for group of atoms
            center = torch.matmul(
                self.coords.view((self.natoms, 3))[atoms].T, self.mass[atoms]
            )
            m_tot = self.mass[atoms].sum()
            com = center / m_tot

        else:
            # only one atom
            atom = int(atoms)
            com = self.coords[3 * atom : 3 * atom + 3]

        return com

    def x(self) -> float:
        """use x axis as cv for numerical examples"""
        self.update_coords()
        self.cv = self.coords[0]
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def y(self) -> float:
        """use y axis as cv for numerical examples"""
        self.update_coords()
        self.cv = self.coords[1]
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

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

        p1 = self._get_com(cv_def[0])
        p2 = self._get_com(cv_def[1])

        # get distance
        r12 = p2 - p1
        self.cv = torch.linalg.norm(r12)

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

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

        p1 = self._get_com(cv_def[0])
        p2 = self._get_com(cv_def[1])
        p3 = self._get_com(cv_def[2])

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
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

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

        p1 = self._get_com(cv_def[0])
        p2 = self._get_com(cv_def[1])
        p3 = self._get_com(cv_def[2])
        p4 = self._get_com(cv_def[3])

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
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

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

    def distorted_distance(self, cv_def: list, r_0: float = 3.0) -> float:
        """distorted distance between two mass centers in range(0, inf) mapped to range(1,0)

        Args:
            cv_def (list):
                distorted distance beteen atoms: [ind0, ind1]
                distorted distance between mass centers: [[ind00, ind01, ...],
                                                          [ind10, ind11, ...]]
            r_0 (float): distance in Angstrom at which the CN function has the value 0.5

        Returns:
            distorted distance (float): computed distance
        """
        self.update_coords()

        if len(cv_def) != 2:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of distance!"
            )

        r_0 /= BOHR_to_ANGSTROM

        p1 = self._get_com(cv_def[0])
        p2 = self._get_com(cv_def[1])

        # get distance
        r12 = p2 - p1
        norm = torch.linalg.norm(r12, dtype=torch.float)
        # to prevent numerical instability
        if norm == r_0:
            d = norm / (r_0 * 1.000001)
        else:
            d = norm / r_0

        self.cv = (1.0 - torch.pow(d, 6)) / (1.0 - torch.pow(d, 12))

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def rmsd(
        self,
        cv_def: Union[str, list],
        method: str = "quaternion",
    ) -> float:
        """rmsd to reference structure

        Args:
            cv_def: path to xyz file with reference structure
                    definition: 'path to xyz' or
                                ['path to reference xyz', [atom indices]]
            method: 'kabsch', 'quaternion' or 'kearsley' algorithm for optimal alignment
                    gradient of kabsch algorithm numerical unstable!

        Returns:
            cv: root-mean-square deviation to reference structure
        """
        self.update_coords()

        if isinstance(cv_def, list):
            atom_indices = cv_def[1]
            cv_def = cv_def[0]
        else:
            atom_indices = None

        if self.reference == None:
            self.reference = read_xyz(cv_def)

        if method.lower() == "kabsch":
            self.cv = kabsch_rmsd(self.coords, self.reference, indices=atom_indices)
        elif method.lower() == "kearsley":
            self.cv = Kearsley().fit(self.coords, self.reference, indices=atom_indices)
        else:  # 'quaternion':
            self.cv = quaternion_rmsd(self.coords, self.reference, indices=atom_indices)

        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def path_progression(
        self,
        cv_def: Union[str, list],
        n_interpol: int = 2,
        method: str = "internal",
    ) -> float:
        """progression along path

        see: Branduardui et al., J. Chem. Phys. (2007); https://doi.org/10.1063/1.2432340
             Leines et. al., Phys. Rev. Lett. (2012), https://link.aps.org/doi/10.1103/PhysRevLett.109.020601

        TODO: Adaptive path updates to converge to MFEP 

        Args:
            cv_def: path to xyz file with reference structure
                    definition: 'path to xyz' or
                                ['path to xyz', [atom indices]]
            n_interpol: number of interpolated images to between two nodes of the reference path
            method: 'gpath' for geometrical path definition based on selected internal distances (Leines et. al.),
                    'internal' for using RMSD of selected internal distances, 
                    'quaternion', 'kabsch' or 'kearsley' for RMSD based on optimal alignment of cartesian coordinates,

        Returns:
            cv: path collective variable
        """
        self.update_coords()

        if isinstance(cv_def, list):
            atom_indices = cv_def[1]
            cv_def = cv_def[0]
        else:
            atom_indices = None

        if self.reference == None:
            images = read_traj(cv_def)
            self.reference = interpolate_coordinates(images, n_interpol=n_interpol)
            if len(self.reference[0]) != len(self.coords):
                raise ValueError(
                    " >>> Error: number of cartesian coordinates has to match reference coordinates"
                )

            if method.lower() == 'gpath':
                #self.close_index = find_closest_points(
                #    self.coords, indices=atom_indices
                #)
                print(
                    f"\n >>> Colvars Info: Geometrical Path CV defined by {len(self.reference)} nodes."
                )
            else:

                print(
                    f"\n >>> Colvars Info: Path CV defined by {len(self.reference)} nodes."
                )
                self.la = 0
                if method.lower() == "internal":
                    print(
                        f" >>> Colvars Info: Using internal coordinates for calculation of RMSD."
                    )
                    if self.reference_internal == None:
                        self.reference_internal = []
                        for image in self.reference:
                            self.reference_internal.append(
                                get_internal_coords(
                                    image, atom_indices,
                                )
                            )
                    for i in range(1, len(self.reference_internal)):
                        self.la += torch.linalg.norm(
                            self.reference_internal[i] - self.reference_internal[i - 1]
                        )

                else:    
                    print(
                        f" >>> Colvars Info: Using {method} algorithm for calculation of RMSD."
                    )
                    for i in range(1, len(self.reference)):
                        self.la += torch.linalg.norm(
                            self.reference[i] - self.reference[i - 1]
                        )

                self.la /= float(len(self.reference))
                self.la = 1.0 / torch.pow(self.la, 2)
                print(
                    f" >>> Colvars Info: Setting lambda parameter to {self.la} 1/Bohr^2."
                )

        M = len(self.reference)

        if method.lower() == 'gpath':
            # use geometrical path definition in internal coordinates
            self.coords_internal = get_internal_coords(
                self.coords, atom_indices,
            )
            if self.reference_internal == None:
                self.reference_internal = []
                for image in self.reference:
                    self.reference_internal.append(
                        get_internal_coords(
                            image, atom_indices,
                        )
                    )

            # find index of nearest image on path
            rmsds = []
            for i, image in enumerate(self.reference_internal):
                rmsds.append(torch.linalg.norm(image - self.coords_internal))
            rmsds = torch.stack(rmsds)
            
            nearest_image = torch.argmin(rmsds)
            if int(nearest_image) == 0:
                nearest_image += 1
            elif int(nearest_image) == len(rmsds):
                nearest_image -= 1

            # compute path cv
            v1 = self.reference_internal[nearest_image] - self.coords_internal
            v2 = self.coords_internal - self.reference_internal[nearest_image - 1]
            v3 = (
                self.reference_internal[nearest_image + 1]
                - self.reference_internal[nearest_image]
            )

            v1_n2 = torch.pow(torch.linalg.norm(v1), 2)
            v2_n2 = torch.pow(torch.linalg.norm(v2), 2)
            v3_n2 = torch.pow(torch.linalg.norm(v3), 2)

            dot13 = torch.dot(v1, v3)
            denom = 2.0 * M * v3_n2

            self.cv = (
                torch.sqrt(torch.pow(dot13, 2) - v3_n2 * (v1_n2 - v2_n2)) / denom
            ) - ((dot13 - v3_n2) / denom)

            if rmsds[nearest_image - 1] < rmsds[nearest_image + 1]:
                self.cv = float(nearest_image) / float(M) - self.cv
            else:
                self.cv = float(nearest_image) / float(M) + self.cv

        else:

            if method.lower() == "internal":
                self.coords_internal = get_internal_coords(
                    self.coords, atom_indices,
                )
            
            num = 0
            denom = 0

            for i, image in enumerate(self.reference):

                if method.lower() == "internal":
                    d = torch.linalg.norm(self.reference_internal[i] - self.coords_internal)
                elif method.lower() == "kabsch":
                    d = kabsch_rmsd(image, self.coords, indices=atom_indices)
                elif method.lower() == "kearsley":
                    d = Kearsley().fit(image, self.coords, indices=atom_indices)
                elif method.lower() == 'quaternion':
                    d = quaternion_rmsd(image, self.coords, indices=atom_indices)
                else:
                    raise ValueError(" >>> Error: invalid method for calculation of RMSD.")

                exp = torch.exp(-self.la * d)
                num += float(i) * exp
                denom += exp

            self.cv = num / denom
            self.cv = self.cv / float(M)  # normalize to range(0,1)

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def path_distance(
        self,
        cv_def: Union[str, list],
        n_interpol: int = 2,
        method: str = "quaternion",
    ) -> float:
        """distance from path 

        see: Branduardui et al., J. Chem. Phys. (2007); https://doi.org/10.1063/1.2432340
             
        Args:
            cv_def: path to xyz file with reference path
                    definition: 'path to xyz' or
                                ['path to xyz', [atom indices]]
            n_interpol: number of interpolated images to between to nodes of path
            method: 'internal' for using RMSD of selected internal distances, 
                    'quaternion', 'kabsch' or 'kearsley' for using optimal alignment of cartesian coordinates,

        Returns:
            cv: distance from path
        """
        self.update_coords()

        if isinstance(cv_def, list):
            atom_indices = cv_def[1]
            cv_def = cv_def[0]
        else:
            atom_indices = None

        if self.reference == None:
            images = read_traj(cv_def)
            self.reference = interpolate_coordinates(images, n_interpol=n_interpol)
            if len(self.reference[0]) != len(self.coords):
                raise ValueError(
                    " >>> Error: number of cartesian coordinates has to match reference coordinates"
                )

            print(
                f"\n >>> Colvars Info: Distance to Path defined by {len(self.reference)} nodes."
            )
            self.la = 0
            if method.lower() == "internal":
                print(
                    f" >>> Colvars Info: Using internal coordinates for calculation of RMSD."
                )
                if self.reference_internal == None:
                    self.reference_internal = []
                    for image in self.reference:
                        self.reference_internal.append(
                            get_internal_coords(
                                image, atom_indices,
                            )
                        )
                for i in range(1, len(self.reference_internal)):
                    self.la += torch.linalg.norm(
                        self.reference_internal[i] - self.reference_internal[i - 1]
                    )

            else:    
                print(
                    f" >>> Colvars Info: Using {method} algorithm for calculation of RMSD."
                )
                for i in range(1, len(self.reference)):
                    self.la += torch.linalg.norm(
                        self.reference[i] - self.reference[i - 1]
                    )

            self.la /= float(len(self.reference))
            self.la = 1.0 / torch.pow(self.la, 2)
            print(
                f" >>> Colvars Info: Setting lambda parameter to {self.la} 1/Bohr^2."
            )

        if method.lower() == "internal":
            self.coords_internal = get_internal_coords(
                self.coords, atom_indices,
            )

        sum = 0
        for i, image in enumerate(self.reference):

            if method.lower() == "internal":
                d = torch.linalg.norm(self.reference_internal[i] - self.coords_internal)
            elif method.lower() == "kabsch":
                d = kabsch_rmsd(image, self.coords, indices=atom_indices)
            elif method.lower() == "kearsley":
                d = Kearsley().fit(image, self.coords, indices=atom_indices)
            elif method.lower() == 'quaternion':
                d = quaternion_rmsd(image, self.coords, indices=atom_indices)
            else:
                raise ValueError(" >>> Error: invalid method for calculation of RMSD.")

            sum += torch.exp(-self.la * d)

        self.cv = - torch.log(sum) / self.la

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def get_cv(self, cv: str, atoms: list, **kwargs) -> Tuple[float, np.ndarray]:
        """get state of collective variable from cv definition of sampling_tools

        Args:
            cv: type of CV
            atoms: indices of atoms

        Returns:
           xi: value of collective variable
           gradient: gradient of collective variable
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
            xi = self.distorted_distance(atoms, **kwargs)
            self.type = None
        elif cv.lower() == "rmsd":
            xi = self.rmsd(atoms)
            self.type = "distance"
        elif cv.lower() == "path":
            xi = self.path_progression(atoms, method='quaternion', **kwargs)
            self.type = None
        elif cv.lower() == "cv_path":
            xi = self.path_progression(atoms, method='internal', **kwargs)
            self.type = None
        elif cv.lower() == "gpath":
            xi = self.path_progression(atoms, method='gpath', **kwargs)
            self.type = None
        elif cv.lower() == "path_distance":
            xi = self.path_distance(atoms, **kwargs)
            self.type = None
        else:
            print(" >>> Error in CV: Unknown coordinate")
            sys.exit(1)

        if self.requires_grad:
            return xi, self.gradient
        else:
            return xi
