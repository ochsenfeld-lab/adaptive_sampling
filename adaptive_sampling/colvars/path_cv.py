import math, time
import torch
import itertools

from ..units import BOHR_to_ANGSTROM, DEGREES_per_RADIAN
from .utils import kabsch_rmsd, quaternion_rmsd

class PathCV:
    """Adaptive Path Collective Variables

    Args:
        guess_path: xyz file with initial path
        active: list of indices of atoms included in PathCV
        n_interpolate: Number of nodes that are added between original nodes by linear interpolation
        metric: Metric for calculation of distance of points (`RMSD`, `kabsch`, `quaternion`, `abs_distance`, `internal`)
        adaptive: if path is adaptive
        update_interval: number of steps between update of adaptive path
        half_life: number of steps til original path weights only half due to updates
        verbose: if verbose information should be printed
    """
    def __init__(
        self, 
        guess_path: str=None, 
        active: list=None, 
        n_interpolate: int=0,
        metric: str="kabsch",
        adaptive: bool=False,
        update_interval: int=100,
        half_life: float=100, 
        verbose: bool=False,
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
        if n_interpolate > 0:
            self.interpolate(n_interpolate)
        self._reparametrize_path()
        self.boundary_nodes = self._get_boundary()

        # accumulators for path update
        if self.adaptive:
            self.sum_weights = torch.zeros(self.nnodes)    
            self.weighted_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]
            self.n_updates = 0 

        if self.verbose:
            print(f" >>> INFO: Initialization of PathCV with {self.nnodes} nodes finished.")

    def calculate_path(self, coords: torch.tensor, distance: bool=False):
        """calculate PathCV according to
        Branduardi, et al., J. Chem. Phys. (2007): https://doi.org/10.1063/1.2432340
        """
        if self.metric.lower() == '2d':
            z = torch.reshape(coords, (self.natoms, 2)).float()
        else:
            z = torch.reshape(coords, (self.natoms, 3)).float()
        z = torch.flatten(z[self.active])

        rmsds = self._get_rmsds_to_path(z)

        # TODO: should the absolute distance or any metric enter lambda?
        la = 1. / torch.square(self._rmsd(self.path[1], self.path[0]))
        #la = 1. / torch.square(torch.linalg.norm(self.path[1] - self.path[0]))
        
        term1 = 0.0
        term2 = 0.0
        for i, rmsd in enumerate(rmsds):
            exp = torch.exp(-la * torch.square(rmsd))
            term1 += float(i) * exp
            term2 += exp

        if not distance:
            # position on path in range [0,1]
            self.path_cv = (1. / (self.nnodes-1)) * (term1 / term2)
        else: 
            # distance from path
            self.path_cv = -1. / la * torch.log(term2)

        if self.adaptive:
            _, coords_nearest = self._get_closest_nodes(z, rmsds)
            self.update_path(z, coords_nearest)

        return self.path_cv

    def calculate_gpath(self, coords: torch.tensor):
        """Calculate geometric PathCV according to
        Leines et al., Phys. Ref. Lett. (2012): https://doi.org/10.1103/PhysRevLett.109.020601
        
        TODO: Fix this, it doesn't work!
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
        vec = z - s
        norm_vec = torch.linalg.norm(vec) 
        return norm_vec

    def update_path(self, z: torch.tensor, q: list):
        """update path nodes to ensure convergence to MFEP

        Args:
            z: reduced coords 
            q: coords of two neighbour nodes of z

        see: Ortiz et al., J. Chem. Phys. (2018): https://doi.org/10.1063/1.5027392
        """
        s = self._project_coords_on_path(z, q)
        w = torch.zeros(self.nnodes)

        xi = math.exp(-math.log(2.) / float(self.half_life))

        for j, _ in enumerate(self.path[1:-1], start=1):
            dist_ij = torch.linalg.norm(self.path[j] - self.path[j+1])
            w = max([0, 1 - torch.linalg.norm(self.path[j] - s) / dist_ij])
            self.sum_weights[j] += xi * w
            self.weighted_dists[j] -= w * (s-z) / torch.linalg.norm(s - z)   # TODO: is s needed?
        self.n_updates += 1
        
        # update path all self.update_interval steps
        if self.n_updates == self.update_interval:
            new_path = self.path.copy()
            for j in range(self.nnodes-1):
                if self.sum_weights[j+1]:
                    new_path[j+1] += (self.weighted_dists[j+1] / self.sum_weights[j+1])
                    new_path[j+1] = new_path[j+1].detach()  # otherwise torch gets vary slow! 
            
            self.path = new_path.copy()
            self._reparametrize_path()
            self.n_updates = 0 
            self.sum_weights = torch.zeros_like(self.sum_weights)
            self.weighted_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]

    def _project_coords_on_path(self, z: torch.tensor, q: list) -> torch.tensor:
        """

        Args:
            z: reduced coords 
            q: coords of two neighbour nodes of z

        Returns:
            s: projection of coords on path 
        """    
        ap = z - q[0]
        ab = q[1] - q[0]
        return q[0] + torch.matmul(ap,ab) / torch.matmul(ab,ab) * ab

    def _get_rmsds_to_path(self, z: torch.tensor) -> list:
        """Calculates RMSD according to chosen distance metric

        Args:
            z: reduced coords 

        Return:
            rmsds: list of RMSDs of z to every node of path
        """
        rmsds = []
        for i in range(self.nnodes):
            rmsds.append(self._rmsd(z, self.path[i]))
            self.path[i] = self.path[i].detach() 
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
        rmsds.insert(0, self._rmsd(z, self.boundary_nodes[0]))
        rmsds.append(self._rmsd(z, self.boundary_nodes[1]))
        closest_idx = sorted(range(len(rmsds)), key=rmsds.__getitem__)
        
        # find closest neighbour of closest node
        for neighbour_idx in closest_idx[1:]:
            if abs(neighbour_idx - closest_idx[0]) == 1:
                break
        
        path_new = self.path.copy()
        path_new.insert(0, self.boundary_nodes[0])
        path_new.append(self.boundary_nodes[1])
        closest_coords = [path_new[closest_idx[0]], path_new[neighbour_idx]]
        
        if self.verbose:
            if abs(closest_idx[0] - closest_idx[1]) != 1:
                print(f" >>> WARNING: Two closest nodes of PathCV ({closest_idx[0]-1, closest_idx[1]-1}) are not neighbours!") 

        return [closest_idx[0]-1, closest_idx[1]-1], closest_coords

    def _reparametrize_path(self, tol = 0.01, max_step: int=1000):
        """ensure equidistant nodes by iteratively placing middle node equidistantly between neighbour nodes,
        while keeping end nodes fixed

        Args:
            tol: tolerance of path distance
            max_step: maximum number of iterations
        """
        step = 0
        while step < max_step:
            step += 1
            
            l_ij = []
            for i, coords in enumerate(self.path[1:], start=1):
                # TODO: should this only use the distance or any RMSD metric?
                l_ij.append(torch.linalg.norm(coords - self.path[i-1]))
                #l_ij.append(self._rmsd(coords, self.path[i-1]))  

            L = sum(l_ij) / (self.nnodes-1)

            path_new = [self.path[0]]
            for i, coords in enumerate(self.path[1:-1]):
                s_i = float(i+1) * L
                vec = coords - self.path[i]
                vec /= torch.linalg.norm(vec)
                path_new.append(self.path[i] + (s_i - sum(l_ij[:i])) * vec) 
            path_new.append(self.path[-1])
            
            l_ij = torch.tensor(l_ij)
            crit = torch.max(torch.abs(l_ij[1:]-l_ij[:1]))
            if crit < tol:
                break
            self.path = path_new.copy()

        if self.verbose:
            if crit < tol:
                print(f" >>> INFO: Reparametrization of Path converged in {step} steps. Max(delta d_ij)={crit:.3f}.")
            else:
                print(f" >>> WARNING: Reparametrization of Path not converged in {max_step} steps. Max(delta d_ij)={crit:.3f}.")

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
        Available metrics to calculate RMSD are: "RMSD", "kabsch", "quaternion"(, "internal")

        Args:
            coords: XYZ coordinates 
            reference: XYZ coordinates of reference

        Returns:
            rmsd: root-mean-square difference between coords and reference
        """
        if self.metric.lower() == "rmsd":
            diff = coords - reference
            rmsd = torch.sqrt(torch.sum(diff * diff) / len(diff))
        
        elif self.metric.lower() == "quaternion":
            rmsd = quaternion_rmsd(coords, reference)
        
        elif self.metric.lower() == "kabsch":
            rmsd = kabsch_rmsd(coords, reference)
        
        elif self.metric.lower() == "abs_distance":
            rmsd = torch.linalg.norm(coords - reference)

        elif self.metric.lower() == "internal": 
            zmatrix_coords = self._cartesians_to_internals(coords)
            zmatrix_reference = self._cartesians_to_internals(reference) 
            diff = zmatrix_coords - zmatrix_reference
            rmsd = torch.sqrt(torch.sum(diff * diff) / len(diff))

        elif self.metric.lower() == "2d":
            diff = coords - reference
            rmsd = torch.sqrt(torch.sum(diff * diff) / 2)
        
        else:
            raise NotImplementedError(
                "Available RMSD metrics are: `RMSD`, `kabsch`, `quaternion`, `abs_distance`, `internal`"
            )

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
                        mol = torch.FloatTensor(list(mol))
                        traj.append(mol)
                        mol = []
                    elif len(words) >= 4:
                        n += 1
                        if metric.lower() == '2d':
                            mol.append([float(words[1]), float(words[2])])
                        else:
                            mol.append([
                                float(words[1]) / BOHR_to_ANGSTROM, 
                                float(words[2]) / BOHR_to_ANGSTROM, 
                                float(words[3]) / BOHR_to_ANGSTROM,
                            ])

            if mol:
                mol = itertools.chain(*mol)
                mol = torch.FloatTensor(list(mol))
                traj.append(mol)
    
        return traj, len(traj), n_atoms

    def interpolate(self, n_interpolate):
        """Add nodes by linear interpolation
        
        Args:
            n_interpolate: Number of interpolated nodes between two original nodes
        """
        path_new = []
        for i in range(self.nnodes - 1):
            a = self.path[i]
            b = self.path[i + 1]
            r = b - a
            d = torch.linalg.norm(r)
            unit_vec = r / d
            step = d / (n_interpolate+1)

            path_new.append(a)
            for j in range(1, n_interpolate+1):
                # linear interpolation
                p = a + unit_vec * j * step
                path_new.append(p)

        path_new.append(b)
        self.path = path_new.copy() 
        self.nnodes = len(self.path)

    def _cartesians_to_internals(self, coords):
        z = torch.reshape(coords, (len(self.active), 3))

        if not hasattr(self, "idx_internal"):
            zmatrix = torch.zeros_like(z)
            
            for i, atom1 in enumerate(z[1:], start=1):
                
                # dist
                zmatrix[i, 0] = torch.linalg.norm(z[i-1] - atom1)

                # angle
                if i > 1:
                    q12 = z[i-2] - z[i-1]
                    q23 = z[i-1] - z[i]

                    q12_n = torch.linalg.norm(q12)
                    q23_n = torch.linalg.norm(q23)

                    q12_u = q12 / q12_n
                    q23_u = q23 / q23_n

                    zmatrix[i, 1] = torch.arccos(torch.dot(-q12_u, q23_u))  
                
                # torsion
                if i > 2:
                    q12 = z[i-2] - z[i-3]
                    q23 = z[i-1] - z[i-2]
                    q34 = z[i-0] - z[i-1]

                    q23_u = q23 / torch.linalg.norm(q23)

                    n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
                    n2 = q34 - torch.dot(q34, q23_u) * q23_u

                    zmatrix[i, 2] = torch.atan2(
                        torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2)
                    )  
        
        return zmatrix

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






