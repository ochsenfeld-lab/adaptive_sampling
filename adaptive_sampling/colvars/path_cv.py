import math, time
import torch
import itertools

from ..units import BOHR_to_ANGSTROM
from .utils import *

class PathCV:
    """Adaptive Path Collective Variables

    Args:
        guess_path: xyz file with initial path
        active: list of indices of atoms included in PathCV
        n_interpolate: Number of nodes that are added between original nodes by linear interpolation
        metric: Metric for calculation of distance of points (`RMSD`, `MSD`, `kabsch`, `quaternion`, `abs_distance`, `internal`, `selected_internal`)
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
        smooth: bool=True,
        smooth_damping: float=0.1,
        coordinates: str="Cartesian",
        metric: str="kabsch",
        ndim: int=3,
        cv_list: list=None,
        adaptive: bool=False,
        update_interval: int=100,
        half_life: float=100, 
        verbose: bool=False,
    ):  
        if guess_path == None:
            raise ValueError(" >>> ERROR: You have to provide a guess path to the PathCV")
        self.guess_path = guess_path
        self.smooth = smooth
        self.smooth_damping = smooth_damping
        self.coordinates = coordinates
        self.metric = metric
        self.cv_list = cv_list
        self.adaptive = adaptive
        self.update_interval = update_interval
        self.half_life = half_life
        self.verbose = verbose
        self.ndim = ndim
        
        # initialized path nodes
        self.path, self.nnodes, _ = read_path(
            self.guess_path, 
            ndim=self.ndim,
        )
        
        self.active = active 
        self._reduce_path()
        
        if n_interpolate > 0:
            self.interpolate(n_interpolate)
        
        self._reparametrize_path(smooth=self.smooth)
        self.boundary_nodes = self._get_boundary(self.path)

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
        z = convert_coordinate_system(
            coords, self.active, coord_system=self.coordinates, ndim=self.ndim
        )
        rmsds = self._get_distance_to_path(z)
        
        la = (
            1. / torch.square(
                self.get_distance(self.path[1], self.path[0], metric=self.metric)
            )
        ).type(torch.float64)
        
        term1 = 0.0
        term2 = 0.0
        for i, rmsd in enumerate(rmsds):
            exp = torch.exp(-la * torch.square(rmsd))
            term1 += float(i) * exp
            term2 += exp

        # avoids numerical inconsistency by never reaching absolute zero
        if abs(term2) < 1.e-15:
            term1 += 1.e-15
            term2 += 1.e-15

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
        z = convert_coordinate_system(
            coords, self.active, coord_system=self.coordinates, ndim=self.ndim
        )
        rmsds = self._get_distance_to_path(z)
        
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
        
        v1 = v1.view(torch.numel(v1))
        v2 = v2.view(torch.numel(v2))
        v3 = v3.view(torch.numel(v3))

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
        z = convert_coordinate_system(
            coords, self.active, coord_system=self.coordinates, ndim=self.ndim
        )
        rmsds = self._get_distance_to_path(z)
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
            dist_ij = self.get_distance(self.path[j], self.path[j+1], metric=self.metric)
            w = max([0, 1 - self.get_distance(self.path[j], s, metric=self.metric) / dist_ij])
            self.sum_weights[j] += xi * w
            self.weighted_dists[j] += w * (z-s) 
        self.n_updates += 1
        
        # update path all self.update_interval steps
        if self.n_updates == self.update_interval:
            new_path = self.path.copy()
            for j in range(self.nnodes-1):
                if self.sum_weights[j+1]:
                    new_path[j+1] += (self.weighted_dists[j+1] / self.sum_weights[j+1])
                    new_path[j+1] = new_path[j+1].detach()

            self.path = new_path.copy()
            self._reparametrize_path(smooth=self.smooth)

            self.n_updates = 0 
            self.sum_weights = torch.zeros_like(self.sum_weights)
            self.weighted_dists = [torch.clone(self.path[i]) for i in range(self.nnodes)]

    def _reparametrize_path(
        self, 
        tol: float=0.01, 
        max_step: int=100,
        smooth: bool=True,
    ):
        """Ensure equidistant nodes by iteratively placing middle nodes equidistant between neighbour nodes,
        while keeping end nodes fixed.
        Smoothing of path can be ensured by projecting position of middle node on line spanned by neighbour nodes

        see: Maragliano et al., J. Chem. Phys. (2006): https://doi.org/10.1063/1.2212942

        Args:
            tol: tolerance of path distance
            max_step: maximum number of iterations
            smooth: if path should be smoothed to remove kinks
            s: damping factor for smoothing in range(0,1)
        """        
        step = 0
        while step < max_step:
            step += 1
            
            if smooth:
                self.path = self._smooth_string(self.path, s=self.smooth_damping)
            
            l_ij = []
            for i, coords in enumerate(self.path[1:], start=1):
                l_ij.append(torch.linalg.norm(coords - self.path[i-1]))

            L = sum(l_ij) / (self.nnodes-1)

            path_new = [self.path[0]]
            for i, coords in enumerate(self.path[1:-1]):
                s_i = float(i+1) * L
                update = s_i - sum(l_ij[:i])

                grad = coords - self.path[i]
                grad /= torch.linalg.norm(grad)

                path_new.append(self.path[i] + update * grad)
            path_new.append(self.path[-1])
            
            self.path = path_new.copy()

            l_ij = torch.tensor(l_ij)
            crit = torch.max(torch.abs(l_ij[1:]-l_ij[:1]))
            if crit < tol:
                break

        if self.verbose:
            if crit < tol:
                print(f" >>> INFO: Reparametrization of Path converged in {step} steps. Max(delta d_ij)={crit:.3f}.")
            else:
                print(f" >>> WARNING: Reparametrization of Path not converged in {max_step} steps. Max(delta d_ij)={crit:.3f}.")

        self.boundary_nodes = self._get_boundary(self.path)

    @staticmethod
    def _smooth_string(path: list, s: float=0.5):
        """Smooth string of nodes 

        Args:
            path: path of nodes to be smoothed
            s: damping factor in range(0,1)
        
        Returns:
            path_new: smoothed path
        """
        path_new = [path[0]]
        for i, coords in enumerate(path[1:-1], start=1):
            path_new.append((1-s) * coords)
            path_new[-1] += s / 2. * (path[i-1] + path[i+1])
        path_new.append(path[-1])
        return path_new

    @staticmethod
    def _get_boundary(path):
        """compute one final lower and upper node by linear interpolation of path
        """
        delta_lower = path[1] - path[0]
        lower_bound = path[0] - delta_lower
        delta_upper = path[-2] - path[-1]
        upper_bound = path[-1] - delta_upper
        return [lower_bound, upper_bound]

    def _get_distance_to_path(self, z: torch.tensor) -> list:
        """Calculates RMSD according to chosen distance metric

        Args:
            z: reduced coords 

        Return:
            rmsds: list of RMSDs of z to every node of path
        """
        rmsds = []
        for i in range(self.nnodes):
            rmsds.append(self.get_distance(z, self.path[i], metric=self.metric))
            self.path[i] = self.path[i].detach()
        return rmsds

    @staticmethod
    def get_distance(
        coords: torch.tensor, 
        reference: torch.tensor,
        metric: str="RMSD"    
    ) -> torch.tensor:
        """Get distance between individual coordinates and reference calculated by `metric`
        Available metrics are: 
            `RMSD`: Root mean square deviation
            `MSD`: Mean square deviation
            `kabsch`: Root mean square deviation of optimally rotated and translated coords
            `KMSD`: Mean square deviation of optimally rotated and translated coords
            `distance`: Absolute distance

        Args:
            coords: coordinates 
            reference: coordinates of reference

        Returns:
            d: distance between coords and reference in `self.metric`
        """
        if metric.lower() == "rmsd":
            d = get_rmsd(coords, reference)
        
        elif metric.lower() == "msd":
            d = get_msd(coords, reference)

        elif metric.lower() == "kmsd":
            coords_fitted, reference_fitted = kabsch_rmsd(
                coords, 
                reference, 
                return_coords=True
            )
            d = get_msd(coords_fitted, reference_fitted)
        
        elif metric.lower() == "kabsch":
            d = kabsch_rmsd(coords, reference)
        
        elif metric.lower() == "distance":
            d = torch.linalg.norm(coords - reference)
        
        else:
            raise NotImplementedError(
                "Available distance metrics are: `RMSD`, `MSD`, `kabsch`, `KMSD`, `distance`"
            )

        return d.type(torch.float64)

    def _project_coords_on_path(self, z: torch.tensor, q: list) -> torch.tensor:
        """

        Args:
            z: reduced coords 
            q: coords of two neighbour nodes of z

        Returns:
            s: projection of coords on path 
        """    
        shape_z = z.shape        # for reshaping results
        ncoords = torch.numel(z) # to flatten inputs

        z = z.view(ncoords)
        q0 = q[0].view(ncoords)
        q1 = q[1].view(ncoords)
        ap = z - q0
        ab = q1 - q0
        return (q0 + torch.matmul(ap,ab) / torch.matmul(ab,ab) * ab).view(shape_z)

    def _get_closest_nodes(self, z: torch.tensor, rmsds: list) -> tuple:
        """get two closest nodes of path

        Args:
            z: reduced coordinates
            rmsds: list of rmsds of z to path nodes

        Returns:
            closest_index: list with indices of two closest nodes to z  
            closest_coords: list with coordinates of two closest nodes to z
        """
        rmsds.insert(0, self.get_distance(z, self.boundary_nodes[0], metric=self.metric))
        rmsds.append(self.get_distance(z, self.boundary_nodes[1], metric=self.metric))
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
            # if path gets highly irregualar this can fail, so print warning
            if abs(closest_idx[0] - closest_idx[1]) != 1:
                print(f" >>> WARNING: Two closest nodes of PathCV ({closest_idx[0]-1, closest_idx[1]-1}) are not neighbours!") 

        return [closest_idx[0]-1, closest_idx[1]-1], closest_coords

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

    def _reduce_path(self):
        """reduce coordinates in path to only active atoms
        """
        path_reduced = []
        for c in self.path:
            path_reduced.append(
                convert_coordinate_system(c, self.active, coord_system=self.coordinates, ndim=self.ndim)
            )
        self.path = path_reduced.copy()
        if self.verbose:
            print(f" >>> INFO: Reduced Path nodes to {torch.numel(self.path[0])} active coordinates.")

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
                dcd.write(BOHR_to_ANGSTROM * coords.detach().numpy().reshape(len(self.active), 3))



