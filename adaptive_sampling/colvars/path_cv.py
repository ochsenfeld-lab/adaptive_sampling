import torch
import itertools

from ..units import BOHR_to_ANGSTROM
from .utils import kabsch_rmsd, quaternion_rmsd

class PathCV:
    """Adaptive Path Collective Variables
    """
    def __init__(
        self, 
        guess_path: str=None, 
        active: list=None, 
        metric: str="2d",
        adaptive: bool=True,
        update_interval: int=100,
        half_life: float=100, 
        verbose: bool=True,
    ):  
        if guess_path == None:
            raise ValueError(" >>> ERROR: You have to provide a guess path to the PathCV")
        self.guess_path = guess_path
        self.metric = metric
        self.adaptive = adaptive
        self.update_interval = update_interval
        self.half_life = half_life
        self.verbose = verbose

        # initialized path nodes
        self.path, self.nnodes, self.natoms = self._read_path(self.guess_path, metric=self.metric)
        self.active = active if active is not None else [i for i in range(self.natoms)]
        self._reduce_path()
        self._reparametrize_path()
        self.boundary_nodes = self._get_boundary()

        # accumulators for path update
        if self.adaptive:
            self.sum_weights = torch.zeros(self.nnodes)    
            self.weighted_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]
            self.n_updates = 0 

        if self.verbose:
            print(f" >>> INFO: Initialization of PathCV with {self.nnodes} nodes finished.")

    def calculate_path(self, coords: torch.tensor):
        """calculate path CV along CV according to

        see: Branduardi, et al., J. Chem. Phys. (2007): https://doi.org/10.1063/1.2432340
        """
        if self.metric.lower() == '2d':
            z = torch.reshape(coords, (self.natoms, 2)).float()
        else:
            z = torch.reshape(coords, (self.natoms, 3)).float()
        z = torch.flatten(z[self.active])

        rmsds = self._get_rmsds_to_path(z)
        z = 0.0
        n = 0.0
        la = 1. / torch.square(self._rmsd(self.path[0], self.path[1]))
        for i, rmsd in enumerate(rmsds):
            exp = torch.exp(-la * torch.square(rmsd))
            z += float(i) * exp
            n += exp

        s = (1. / (self.nnodes -1)) * (z / n)
        z = -1. / la * torch.log(n)

        if self.adaptive:
            _, coords_nearest = self._get_closest_nodes(z, rmsds)
            self.update_path(z, coords_nearest) 

        return s, z

    def calculate_gpath(self, coords: torch.tensor):
        """calculate path CV along CV according to

        see: Leines et al., Phys. Ref. Lett. (2012): https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.020601
        """
        if self.metric.lower() == '2d':
            z = torch.reshape(coords, (self.natoms, 2)).float()
        else:
            z = torch.reshape(coords, (self.natoms, 3)).float()
        z = torch.flatten(z[self.active])
        
        rmsds = self._get_rmsds_to_path(z)
        index_nearest, coords_nearest = self._get_closest_nodes(z, rmsds)

        # calculate projection of coords on path
        v1 = coords_nearest[0] - z       
        
        if index_nearest[0] > 0:
            v2 = z - self.path[index_nearest[0]-1] 
        else:
            v2 = z - self.boundary_nodes[0] 
        
        if index_nearest[0] < self.nnodes-1:
            v3 = self.path[index_nearest[0]+1] - coords_nearest[0]
        else:
            v3 = self.boundary_nodes[1] - coords_nearest[0]
        
        v1_sqnorm = torch.square(torch.linalg.norm(v1))
        v2_sqnorm = torch.square(torch.linalg.norm(v2))
        v3_sqnorm = torch.square(torch.linalg.norm(v3)) 
        v1v3      = torch.matmul(v1, v3)
        denom     = 2. * self.nnodes * v3_sqnorm
        sign      = 1. if index_nearest[0] < index_nearest[1] else -1.

        term0 = float(index_nearest[0]) / float(self.nnodes)
        term1 = torch.sqrt(torch.square(v1v3)-v3_sqnorm*(v1_sqnorm - v2_sqnorm)) / denom
        term2 = (v1v3 - v3_sqnorm) / denom

        self.path_cv = term0 + sign * term1 - term2

        if self.adaptive:
            self.update_path(z, coords_nearest)
        
        return self.path_cv

    def tube_potential(self, coords: torch.tensor) -> torch.tensor:
        """Constrain dynamics perpendicular to path with tube like potential

        Args:
            coords: xyz coordinates of system
        
        Returns:
            d: distance to projection of coords on path
        """
        if self.metric.lower() == '2d':
            z = torch.reshape(coords, (self.natoms, 2)).float()            
        else:
            z = torch.reshape(coords, (self.natoms, 3)).float()
        z = torch.flatten(z[self.active])
        rmsds = self._get_rmsds_to_path(z)
        _, q = self._get_closest_nodes(z, rmsds)
        s = self._project_coords_on_path(z, q)
        return torch.linalg.norm(s - z)

    def update_path(self, z: torch.tensor, q: list):
        """update path nodes

        Args:
            z: reduced coords 

        see: Ortiz et al., J. Chem. Phys. (2018): https://doi.org/10.1063/1.5027392
        """
        s = self._project_coords_on_path(z, q)
        #norm_s = torch.linalg.norm(s)
        dist_z = torch.linalg.norm(s-z)
        w = torch.zeros(self.nnodes)
        for j, _ in enumerate(self.path[1:-1], start=1):
            dist_ij = torch.linalg.norm(self.path[j] - self.path[j+1])
            w[j] = max([0, 1 - torch.linalg.norm(self.path[j] - s) / dist_ij])
            self.weighted_dists[j] += w[j] * dist_z #* s / norm_s
        self.sum_weights += w 
        self.n_updates += 1

        # update path all self.update_interval steps
        if self.n_updates == self.update_interval:
            for j, p in enumerate(self.path[1:-1], start=1):
                if self.sum_weights[j] != 0:
                    p += self.weighted_dists[j] / self.sum_weights[j]   
            self._reparametrize_path()
            self.n_updates = 0 
            self.sum_weights = torch.zeros_like(self.sum_weights)
            self.weighted_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]

    def _project_coords_on_path(self, z: torch.tensor, q: list) -> torch.tensor:
        """

        Args:
            z: reduced coords 
            rmsds: list of RMSDs of z to path nodes

        Returns:
            s: projection of coords on path 
        """
        return q[0] + (
            torch.matmul(z-q[0], q[1]-q[0])*(q[1]-q[0]) / (torch.linalg.norm(z-q[0])*torch.linalg.norm(q[1]-q[0]))
        )

    def _get_rmsds_to_path(self, z: torch.tensor):
        rmsds = []
        for i in range(self.nnodes):
            rmsds.append(self._rmsd(z, self.path[i]))
        return rmsds

    def _get_closest_nodes(self, z: torch.tensor, rmsds: list) -> tuple:
        """get two closest nodes of path

        Args:
            z: reduced coordinates
            rmsds: list of rmsds of z to path nodes

        Returns:
            closest_index: list with indices of two closest nodes to z  
            closest_coords: list with coordinates of two closest nodes to z
        """
        closest_index = rmsds.index(min(rmsds))
        s0 = self.path[closest_index]
        if closest_index == 0:
            neighbour_rmsds = [self._rmsd(z, self.boundary_nodes[0]), rmsds[1]]
            second_index = neighbour_rmsds.index(min(neighbour_rmsds))
            s1 = self.boundary_nodes[0] if second_index == 0 else self.path[1]
            second_index = -1 if second_index == 0 else 1
        elif closest_index == self.nnodes-1:
            neighbour_rmsds = [self._rmsd(z, self.boundary_nodes[1]), rmsds[-2]]
            second_index = neighbour_rmsds.index(min(neighbour_rmsds))
            s1 = self.boundary_nodes[1] if second_index == 0 else self.path[-2]            
            second_index = self.nnodes if second_index == 0 else self.nnodes-2
        else:
            neighbour_rmsds = [rmsds[closest_index-1], rmsds[closest_index+1]]
            second_index = rmsds.index(min(neighbour_rmsds))
            s1 = self.path[second_index]

        return [closest_index, second_index], [s0, s1]

    def _reparametrize_path(self, tol: float=0.01, max_steps: int=100):
        """ensure equidistant nodes by iteratively placing middle node equidistantly between neighbour nodes,
        while keeping end nodes fixed

        Args:
            tol: tolerance of path distance
        """
        new_path = self.path.copy()

        step = 0 
        old_L = 9999
        while True:
            
            # relocate path nodes by linear interpolation between neighbours
            for m, coords in enumerate(new_path[1:-1], start=1):
                vector = new_path[m+1] - new_path[m-1]
                new_path[m] = new_path[m-1] + vector / 2.
                
            # get length of path
            L = []
            for m, coords in enumerate(new_path):
                L.append(self._rmsd(coords, new_path[m-1]))
            
            # converged if max absolute difference < tol
            L = torch.as_tensor(L)
            crit = torch.max(torch.abs(L - old_L))
            if crit < tol or step == max_steps:
                if self.verbose:
                    if step < max_steps:
                        print(f' >>> INFO: Reparametrization of PathCV converged in {step} iterations.')
                    else:
                        print(f' >>> WARNING: Reparametrization of PathCV not converged in {step} iterations.')
                break
            old_L = torch.clone(L)
            step += 1

        self.path = new_path.copy()
        self.boundary_nodes = self._get_boundary()

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
        if self.metric.lower() == "rmsd":
            coords = torch.reshape(coords, (self.natoms, 3))
            reference = torch.reshape(reference, (self.natoms, 3))
            diff = coords - reference
            rmsd = torch.sqrt(torch.sum(diff * diff) / len(coords))
        elif self.metric.lower() == 'quaternion':
            rmsd = quaternion_rmsd(coords, reference)
        elif self.metric.lower() == "kabsch":
            rmsd = kabsch_rmsd(coords, reference)
        elif self.metric.lower() == 'internal':
            # TODO: implement more robust RMSD by using selected internal coordinates
            raise NotImplementedError("Available RMSD metrics are: `kabsch`, `quaternion`")
        elif self.metric.lower() == "2d":
            coords = torch.reshape(coords, (self.natoms, 2))
            reference = torch.reshape(reference, (self.natoms, 2))
            diff = coords - reference
            rmsd = torch.sqrt(torch.sum(diff * diff) / len(coords))
        else:
            raise NotImplementedError("Available RMSD metrics are: `kabsch`, `quaternion`")
        return rmsd

    @staticmethod
    def _read_path(filename: str, metric: str=None) -> tuple:
        """Read cartesian coordinates of trajectory from file (*.xyz)
        
        Args:
            xyz_name (str): file-name of xyz-file
        
        Returns:
            traj: list of torch arrays containing xyz coordinates of nodes
            nnodes: number of nodes in path
            natoms: number of atoms of system
        """
        if filename[-3:] == "dcd":
            # TODO: Read guess path from dcd trajectory file
            pass
        else:
            with open(filename, "r") as xyzf:
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
                        if metric.lower() == '2d':
                            mol.append([float(words[1]), float(words[2])])
                        else:
                            mol.append([float(words[1]), float(words[2]), float(words[3])])

            if mol:
                mol = itertools.chain(*mol)
                mol = torch.FloatTensor(list(mol)) / BOHR_to_ANGSTROM
                traj.append(mol)
    
        return traj, len(traj), n_atoms

    def _reduce_path(self):
        """reduce coordinates in path to only active atoms
        """
        if len(self.active) < self.natoms and self.metric.lower() != '2d':
            path_reduced = []
            for c in self.path:
                coords = torch.reshape(c, (self.natoms, 3)).float()
                path_reduced.append(torch.flatten(coords[self.active]))
            self.path = path_reduced.copy()

            if self.verbose:
                print(f" >>> INFO: Reduced PathCV to {len(self.active)} active atoms.")

        elif len(self.active) > self.natoms:
            raise ValueError(f"Number of active atoms > number of atoms!")

    def write_path(self, filename: str="path_cv.npy"):
        """write nodes of PathCV to dcd trajectory file

        Args:
            filename
        """
        if filename[-3:] == "npy":
            import numpy as np
            path_tmp = []
            for coords in self.path:
                path_tmp.append(coords.detach().numpy())
            np.save(filename, path_tmp, allow_pickle=True)
        else:
            import mdtraj
            dcd = mdtraj.formats.DCDTrajectoryFile(filename, 'w', force_overwrite=True)
            for coords in self.path:
                dcd.write(BOHR_to_ANGSTROM * coords.detach().numpy().reshape(self.natoms, 3))






