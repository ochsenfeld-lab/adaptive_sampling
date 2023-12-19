import sys
import torch
import numpy as np

from typing import Union, Tuple
from ..interface.sampling_data import MDInterface
from ..units import *
from .utils import *


class CV:
    """Class for Collective Variables

    Automatic computation of gradients is implemented using the autograd differentiation engine of Torch

    Args:
        the_mol: Molecule Object containing masses (natoms) and coords (3*natoms)
        requires_grad: if True, partial derivatives of CVs with respect
                                to atom coordinats are computed
                                and saved to self.gradient
    """

    def __init__(self, the_mol: MDInterface, requires_grad: bool = False, device='cpu'):

        self.the_mol = the_mol
        self.requires_grad = requires_grad
        self.device = device

        if self.requires_grad:
            torch.autograd.set_detect_anomaly(True)

        md_state = self.the_mol.get_sampling_data()

        self.mass = torch.from_numpy(md_state.mass).float()
        self.coords = torch.from_numpy(md_state.coords.ravel()).float()
        self.mass = self.mass.to(device)
        self.coords = self.coords.to(device)
        
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
        self.coords = self.coords.float().to(self.device)
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
                list of CVs with prefactors: [[type0, fac0, [ind00, ind01]],
                                              [type1, fac1, [ind10, ind11, ind12]],
                                               ...]
                type can be 'distance', 'angle', 'torsion' or 'coordination_number'

        Returns:
            cv (float): linear combination of distances/angles/dihedrals
        """
        self.update_coords()

        self.lc_contribs = []
        gradient = np.zeros(3 * self.natoms, dtype=float)

        for cv in cv_def:

            if cv[0].lower() == 'distance':
                x = self.distance(cv[2])

            elif cv[0].lower() == 'distance_min':
                x = self.distance_min(cv[2])

            elif cv[0].lower() == 'angle':
                x = self.angle(cv[2])

            elif cv[0].lower() == 'torsion':
                x = self.torsion(cv[2])

            elif cv[0].lower() == 'coordination_number':
                x = self.coordination_number(cv[2])

            else:
                raise ValueError(
                    "CV ERROR: Invalid CV in definition of linear combination!"
                )

            self.lc_contribs.append(cv[1] * x)
            if self.requires_grad:
                gradient += cv[1] * self.gradient

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

    def coordination_number(self, cv_def: list) -> float:
        """Coordination Number

        Args:
            cv_def (list): 
                [[idx0, idxA], [idx1, idxA], ..., r_eq, exp_nom, exp_denom]
                with indices of cordinated atoms ind0, ind1, ..., 
                distance r_0 in Angstrom (bonds smaller than r_0 are coordinated), 
                and exponent of the nominator and denominator exp_nom and exp_denom
        """
        exp_denom = int(cv_def[-1])
        exp_nom = int(cv_def[-2])
        r_0 = float(cv_def[-3]) / BOHR_to_ANGSTROM

        self.cv = 0.0
        for atoms in cv_def[:-3]:
            p1 = self._get_com(atoms[0])
            p2 = self._get_com(atoms[1])
            r12 = torch.linalg.norm(p2 - p1)

            # for numerical stability
            if abs(r12-r_0) < 1.e-6:
                r = r12 / (r_0 * 1.000001)
            else:
                r = r12 / r_0

            nom   = 1. - torch.pow(r, exp_nom)
            denom = 1. - torch.pow(r, exp_denom)     
            self.cv += nom / denom

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def distance_min(self, cv_def: list) -> float:
        """distorted distance between two mass centers in range(0, inf) mapped to range(1,0)

        Args:
            cv_def (list):
                list of distances beteen atoms: [[ind0, ind1], [], ...]

        Returns:
            distorted distance (float): computed distance
        """
        self.update_coords()
        
        p1 = self._get_com(cv_def[0])
        p2 = self._get_com(cv_def[1])

        # get distances
        dists = []
        for atoms in cv_def:
            p1 = self._get_com(atoms[0])
            p2 = self._get_com(atoms[1])
            dists.append(torch.linalg.norm(p2 - p1, dtype=torch.float))

        self.cv = min(dists)

        # get forces
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)
    
    def electrostatic_potential(
        self, 
        cv_def: list,
    ):
        """Electrostatic potential on spezific Atom. Environmental CV to treat reorganization of polar solvent or protein sites  
        needs a file called `charges.npy` that contains the charges of all atoms in cv_def
 
        Args:
            cv_def: list with involved atoms, the first element defines the atom where the potential is calculated

        Returns:
            cv: electroststic potential in a.u.
        """
        try:
            charges = np.load("charges.npy")
        except:
            raise ValueError("CV ERROR: Could not find charges for electrostatic potential in charges.npy")

        if len(cv_def) != len(charges)+1:
            raise ValueError(
                "CV ERROR: Number of charges for electrostatic potential has to match number of Atoms!"
            )
        self.update_coords()

        A = self._get_com(cv_def[0])
        self.cv = 0
        for i, atom in enumerate(cv_def[1:]):
            if atom != cv_def[0]:
                B = self._get_com(atom)
                self.cv += charges[i] / torch.linalg.norm(B-A, dtype=torch.float)

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
        else:  # 'quaternion':
            self.cv = quaternion_rmsd(self.coords, self.reference, indices=atom_indices)

        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()

        return float(self.cv)

    def path(self, cv_def: dict, method: str="gpath") -> float:
        """Adaptive path collective variable

        Args:
            cv_def: dictionary of parameters for PathCV
        """
        if not hasattr(self, 'pathcv'):
            from .path_cv import PathCV
            self.pathcv = PathCV(**cv_def)
        
        self.update_coords()

        if method == 'gpath':
            self.cv = self.pathcv.calculate_gpath(self.coords)
        else:
            self.cv = self.pathcv.calculate_path(self.coords)
       
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, self.coords, allow_unused=True
            )[0]
            self.gradient = self.gradient.detach().numpy()
        return float(self.cv)

    def path_z(self, pathcv: object) -> float:
        """Get z component of path cv (distance to path)
        only available if `self.path` was called first to calculate PathCV

        Args:
            pathcv: PathCV object that contains path_z 
        """
        if not hasattr(pathcv, "path_z"):
            raise ValueError(" >>> ERROR: `pathcv` has to `requires_z`!")
        
        self.cv = pathcv.path_z
        self.gradient = pathcv.grad_z
        return float(self.cv)

    def cec(self, pt_def: dict) -> float:
        """ Modified Center of Excess Charge for Proton Transfer

        Args: 
            pt_def: definition of proton transfer coordinate,
                must contain `cv_def`, which specifies indices of contributing atoms
        """
        if not hasattr(self, 'pt_cv'):
            from .proton_transfer import PT
            pt_cv = PT(
                r_sw   = pt_def.get("r_sw", 1.4),
                d_sw   = pt_def.get("d_sw", 0.05),
                n_pair = pt_def.get("n_pair", 15), 
                requires_grad=self.requires_grad,
            )
            
        self.update_coords()
        self.cv = pt_cv.cec(
            self.coords, 
            pt_def["proton_idx"],
            pt_def["heavy_idx"],
            pt_def["heavy_weights"],
            pt_def["ref_idx"],
            pair_def=pt_def.get("pair_def", []),
            modified=pt_def.get("modified", True),
        )
        if self.requires_grad:
            self.gradient = pt_cv.gradient

        return float(self.cv)
        
    def gmcec(self, pt_def: dict) -> float:
        """ Generalized Modified Center of Excess Charge for Proton Transfer

        Args: 
            pt_def: dict with definition of proton transfer coordinate,
                must contain `proton_idx`, `heavy_idx`, `heavy_weights` and `ref_idx` 
        """
        if not hasattr(self, 'pt_cv'):
            from .proton_transfer import PT
            pt_cv = PT(
                r_sw   = pt_def.get("r_sw", 1.4),
                d_sw   = pt_def.get("d_sw", 0.05),
                n_pair = pt_def.get("n_pair", 15), 
                requires_grad = self.requires_grad,
            )
        
        self.update_coords()
        self.cv = pt_cv.gmcec(
            self.coords, 
            pt_def["proton_idx"],
            pt_def["heavy_idx"],
            pt_def["heavy_weights"],
            pt_def["ref_idx"],
            pair_def=pt_def.get("pair_def", []),
            mapping=pt_def.get("mapping", "default"),
        )
        if self.requires_grad:
            self.gradient = pt_cv.gradient

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
        elif cv.lower() == "minimum_distance":
            xi = self.distance_min(atoms)
            self.type = "distance"
        elif cv.lower() == "lin_comb_dists":
            xi = self.linear_combination(atoms)
            self.type = "distance"
        elif cv.lower() == "lin_comb_dists_min":
            xi0 = atoms[0] * self.distance(atoms[1])
            forces0 = atoms[0] * np.copy(self.gradient)
            xi = xi0 + atoms[2] * self.distance_min(atoms[3])
            self.cv = float(xi)
            forces1 = atoms[2] * np.copy(self.gradient)
            self.gradient = forces0 + forces1
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
        elif cv.lower() == "coordination_number":
            xi = self.coordination_number(atoms)
            self.type = None
        elif cv.lower() == "rmsd":
            xi = self.rmsd(atoms)
            self.type = "distance"
        elif cv.lower() == "path":
            xi = self.path(atoms, method="path")
            self.type = "2d" if self.pathcv.ndim == 2 else None 
        elif cv.lower() == "gpath":
            xi = self.path(atoms, method="gpath")
            self.type = "2d" if self.pathcv.ndim == 2 else None 
        elif cv.lower() == "path_z":
            xi = self.path_z(atoms)
            self.type = "2d" if atoms.ndim == 2 else None 
        elif cv.lower() == "cec" or cv.lower() == "mcec":
            xi = self.cec(atoms)
            self.type = "distance"
        elif cv.lower() == "gmcec":
            xi = self.gmcec(atoms)
            self.type = "distance" if atoms.get("mapping", None) not in ["f_sw", "fraction"] else None
        else:
            print(" >>> Error in CV: Unknown Collective Variable")
            sys.exit(1)

        if self.requires_grad:
            return xi, self.gradient
        else:
            return xi
