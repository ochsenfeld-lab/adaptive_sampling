import os, sys
import torch
import itertools

from ..units import BOHR_to_ANGSTROM
from .utils import kabsch_rmsd, quaternion_rmsd

class PathCV:

    def __init__(self, guess_path: str=None, active: list=None, metric: str="kabsch"):
        self.guess_path = guess_path
        self.metric = metric
        self.path, self.nnodes, self.natoms = self._read_path(self.guess_path)
        self.nnodes = len(self.path)
        self.active = active if active is not None else [i for i in range(self.natoms)]
        self.n_avgflux = [0 for _ in range(self.nnodes)]
        self.avgflux   = [None for _ in range(self.nnodes)]
        self.boundary_nodes = self._get_boundary()

    def calculate(self, coords: torch.tensor):
        """calculate path CV along CV according to

        see: Leines et al., Phys. Ref. Lett. (2012): https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.020601
        """
        rmsds = []
        for i in range(self.nnodes):
            rmsds.append(self._rmsd(coords, self.path[i]))

        min_rmsd = min(rmsds)
        index_min = rmsds.index(min_rmsd)

        # update average transition flux <z> at current node along the path
        self.n_avgflux[index_min] += 1
        if self.n_avgflux[index_min] == 1:
            self.avgflux[index_min] = coords
        else:
            delta = coords - self.avgflux[index_min]
            self.avgflux[index_min] += delta / self.n_avgflux[index_min] 

        # calculate projection of coords on path
        v1 = self.path[index_min] - self.avgflux[index_min]
        if index_min > 0:
            v2 = self.avgflux[index_min] - self.path[index_min-1]
        else:
            v2 = self.avgflux[index_min] - self.boundary_nodes[0]
        if index_min < self.nnodes-1:
            v3 = self.path[index_min+1] - self.path[index_min]
        else:
            v3 = self.boundary_nodes[1] - self.path[-1]
        
        # sign changes if coords are left or right if nearest nodes
        if index_min == 0:
            rmsd_lower = self._rmsd(coords, self.boundary_nodes[0])
            sign = -1. if rmsd_lower < rmsds[1] else 1.
        elif index_min == self.nnodes-1:
            rmsd_upper = self._rmsd(coords, self.boundary_nodes[1])
            sign = -1. if rmsds[-1] < rmsd_upper else 1.
        else: 
            sign = -1. if rmsds[index_min-1] < rmsds[index_min+1] else 1.

        v1_sqnorm = torch.square(torch.linalg.norm(v1))
        v2_sqnorm = torch.square(torch.linalg.norm(v2))
        v3_sqnorm = torch.square(torch.linalg.norm(v3)) 
        v1v3      = torch.matmul(v1, v3)
        denom     = 2. * self.nnodes * v3_sqnorm
        
        term0 = float(index_min) / float(self.nnodes)
        term1 = torch.sqrt(torch.square(v1v3)-v3_sqnorm*(v1_sqnorm - v2_sqnorm)) / denom
        term2 = (v1v3 - v3_sqnorm) / denom

        self.path_cv = term0 + sign * term1 - term2
        return self.path_cv

    def update_path(self):
        """update path nodes
        """
        pass

    def _reparametrize_path(self):
        """ensure equdistant nodes by reparametrization
        """
        pass

    def _get_boundary(self):
        """compute one final lower and upper node by linear interpolation of path
        """
        delta_lower = self.path[1] - self.path[0]
        lower_bound = self.path[0] - delta_lower

        delta_upper = self.path[-2] - self.path[-1]
        upper_bound = self.path[-1] - delta_upper
        return [lower_bound, upper_bound]

    def _rmsd(
        self, 
        coords: torch.tensor, 
        reference:torch.tensor, 
    ) -> torch.tensor:
        """Get RMSD between individual nodes and the current coordinates

        Args:
            coords: XYZ coordinates 
            reference: Reference coordinates

        Returns:
            rmsd: root-mean-square difference between coords and reference
        """
        if self.metric.lower() == "kabsch":
            rmsd = kabsch_rmsd(coords, reference, indices=self.active)
        elif self.metric.lower() == 'quaternion':
            rmsd = quaternion_rmsd(coords, reference, indices=self.active)
        elif self.metric.lower() == 'internal':
            # TODO: implement more robust RMSD by using internal coordinates
            raise NotImplementedError("Available RMSD metrics are: `kabsch`, `quaternion`")
        else:
            raise NotImplementedError("Available RMSD metrics are: `kabsch`, `quaternion`")

        return rmsd

    @staticmethod
    def _read_path(filename: str) -> tuple:
        """Read cartesian coordinates of trajectory from file (*.xyz)
        
        Args:
            xyz_name (str): file-name of xyz-file
        
        Returns:
            traj: list of torch arrays containing xyz coordinates of nodes
            nnodes: number of nodes in path
            natoms: number of atoms of system
        """
        with open(filename) as xyzf:
            traj = []
            mol = []
            n = 0
            for i, line in enumerate(xyzf):
                words = line.strip().split()
                if i == 0:
                    n_atoms = int(words[0])
                elif n == n_atoms:
                    n = 0
                    mol = itertools.chain(*mol)
                    mol = torch.FloatTensor(list(mol)) / BOHR_to_ANGSTROM
                    traj.append(mol)
                    mol = []
                elif len(words) >= 4:
                    n += 1
                    mol.append([float(words[1]), float(words[2]), float(words[3])])

        if mol:
            mol = itertools.chain(*mol)
            mol = torch.FloatTensor(list(mol)) / BOHR_to_ANGSTROM
            traj.append(mol)

        return traj, len(traj), n_atoms

    




