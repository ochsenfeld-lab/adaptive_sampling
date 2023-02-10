import math, time
import torch
import itertools

from ..units import BOHR_to_ANGSTROM
from .utils import *

class PathCV:
    """Adaptive Path Collective Variables

    Args:
        guess_path: xyz file with initial path
        active: list of atom indices or CVs included in PathCV 
        n_interpolate: Number of nodes that are added between original nodes by linear interpolation
        smooth_damping: float in range [0,1] (0: no smoothing, 1: linear interpolation between neighbours)
        coordinate_system: coordinate system for calculation of Path CV
            available: `cartesian`, `zmatrix` or `cv_space`
        metric: Metric for calculation of distance of points
            available: `RMSD`, `MSD`, `kabsch`, `KMSD`, `distance`
        adaptive: if adaptive, path converges to averge CV density perpendicular to path
        update_interval: number of steps between update of adaptive path
        half_life: number of steps til weight of original path is half due to updates
        ndim: number of dimensions (2 for 2D test potentials, 3 else)
        verbose: if verbose information should be printed
    """
    def __init__(
        self, 
        guess_path: str=None, 
        active: list=None, 
        n_interpolate: int=0,
        smooth_damping: float=0.1,
        reparam_steps: int=1,
        coordinate_system: str="Cartesian",
        metric: str="Kabsch",
        adaptive: bool=False,
        update_interval: int=100,
        half_life: float=100, 
        ndim: int=3,
        verbose: bool=False,
    ):  
        if guess_path == None:
            raise ValueError(" >>> ERROR: You have to provide a guess path to the PathCV")
        self.guess_path = guess_path
        self.smooth_damping = smooth_damping
        self.reparam_steps = reparam_steps
        self.coordinates = coordinate_system
        self.metric = metric
        self.adaptive = adaptive
        self.update_interval = update_interval
        self.half_life = half_life
        self.verbose = verbose
        self.ndim = ndim
        
        # initialized path nodes
        self.path, self.nnodes = read_path(
            self.guess_path, 
            ndim=self.ndim,
        )
        
        self.active = active 
        self._reduce_path()
        self._interpolate(n_interpolate)
        
        self._reparametrize_path(max_step=self.reparam_steps)
        self.boundary_nodes = self._get_boundary(self.path)

        # accumulators for path update
        if self.adaptive:
            self.sum_weights = torch.zeros(self.nnodes)    
            self.weighted_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]
            self.n_updates = 0 

        if self.verbose:
            print(f" >>> INFO: Initialization of PathCV with {self.nnodes} nodes finished.")

    def calculate_path(self, coords: torch.tensor, distance: bool=False) -> torch.tensor:
        """calculate PathCV according to
        Branduardi, et al., J. Chem. Phys. (2007): https://doi.org/10.1063/1.2432340

        Args:
            coords: cartesian coordinates
            distance: if True, get orthogonal distance to path instead of progress
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

    def calculate_gpath(self, coords: torch.tensor, distance: bool=False) -> torch.tensor:
        """Calculate geometric PathCV according to
        Leines et al., Phys. Ref. Lett. (2012): https://doi.org/10.1103/PhysRevLett.109.020601
        
        Args:
            coords: cartesian coordinates
            distance: if True, get orthogonal distance to path instead of progress
        """
        z = convert_coordinate_system(
            coords, self.active, coord_system=self.coordinates, ndim=self.ndim
        )
        dists = self._get_distance_to_path(z)
        idx_nodemin, coords_nodemin = self._get_closest_nodes(z, dists)

        # add boundary nodes to `self.path`
        self.path.insert(0, self.boundary_nodes[0])
        self.path.append(self.boundary_nodes[1])
        idx_nodemin = [i+1 for i in idx_nodemin]

        isign = idx_nodemin[0] - idx_nodemin[1]
        if isign > 0:
            isign = 1
        elif isign < 0:
            isign = -1
        
        # TODO: can idx_nodemin[0] be one of the boundry nodes
        #if idx_nodemin[0] == 0:
        #    idx_nodemin[0] += 1
        #elif idx_nodemin[0] ==self.nnodes+1:
        #    idx_nodemin[0] -= 1

        idx_nodemin[1] = idx_nodemin[0] - isign
        idx_nodemin.append(idx_nodemin[0] + isign)
        
        # compute some vectors
        v1 = self.path[idx_nodemin[0]] - z
        v3 = z - self.path[idx_nodemin[1]] 
        if (idx_nodemin[2] < 0) or (idx_nodemin[2] >= len(self.path)):
            v2 = self.path[idx_nodemin[0]] - self.path[idx_nodemin[1]]
        else:
            v2 = self.path[idx_nodemin[2]] - self.path[idx_nodemin[0]]
        
        # remove boundary nodes from `self.path`
        del self.path[0]
        del self.path[-1]

        # actual computation of path cv
        v1 = v1.view(torch.numel(v1))
        v2 = v2.view(torch.numel(v2))
        v3 = v3.view(torch.numel(v3))

        v1v1 = torch.matmul(v1, v1)
        v1v2 = torch.matmul(v1, v2)
        v2v2 = torch.matmul(v2, v2) 
        v3v3 = torch.matmul(v3, v3)

        root = torch.sqrt(torch.square(v1v2) - v2v2 * (v1v1 - v3v3))
        dx = 0.5 * (( root - v1v2) / v2v2 - 1.)
        self.path_cv = ((idx_nodemin[0]-1) + isign * dx) / (self.nnodes-1)

        if self.adaptive:
            self.update_path(z, coords_nodemin)

        return self.path_cv

    def path_distance(self, coords: torch.tensor) -> torch.tensor:
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
        norm_vec = torch.linalg.norm(z - self._project_coords_on_line(z, q)) 
        return norm_vec

    def update_path(self, z: torch.tensor, q: list):
        """update path nodes to ensure smooth convergence to the MFEP
        
        see: Ortiz et al., J. Chem. Phys. (2018): https://doi.org/10.1063/1.5027392

        Args:
            z: reduced coords 
            q: coords of two neighbour nodes of z
        """
        s = self._project_coords_on_line(z, q)
        xi = math.exp(-math.log(2.) / float(self.half_life))
        dist_ij = self.get_distance(self.path[0], self.path[1], metric=self.metric)
        for j, _ in enumerate(self.path[1:-1], start=1):
            w = max([0, 1 - self.get_distance(self.path[j], s, metric=self.metric) / dist_ij])
            self.sum_weights[j] += xi * w
            self.weighted_dists[j] += w * (z-s)
            self.sum_weights[j] = self.sum_weights[j].detach()
            self.weighted_dists[j] = self.weighted_dists[j].detach()
        self.n_updates += 1
        
        # update path all `self.update_interval` steps
        if self.n_updates == self.update_interval:
            new_path = self.path.copy()
            for j in range(self.nnodes-1):
                if self.sum_weights[j+1]:
                    new_path[j+1] += (self.weighted_dists[j+1] / self.sum_weights[j+1])
                    new_path[j+1] = new_path[j+1].detach()

            self.path = new_path.copy()
            self._reparametrize_path(max_step=self.reparam_steps)

            self.n_updates = 0 
            self.sum_weights = torch.zeros_like(self.sum_weights)
            self.weighted_dists = [torch.zeros_like(self.path[i]) for i in range(self.nnodes)]


    def _reparametrize_path(
        self, 
        tol: float=0.1, 
        max_step: int=10,
        smooth: bool=True,
    ):
        if smooth:
            self.path = self._smooth_string(self.path, s=self.smooth_damping)
 
        # get length of path
        L = [0]
        sumlen = [0]
        for i, coords in enumerate(self.path[1:], start=1):
            L.append(torch.linalg.norm(coords - self.path[i-1]))
            sumlen.append(sumlen[-1] + L[-1])

        prevsum, iter = 0, 0
        while abs(sumlen[-1]-prevsum) > tol and iter <= max_step:
            prevsum = sumlen[-1]

            # cumulative target distance between nodes
            sfrac = []
            for i in range(self.nnodes):
                sfrac.append(i * sumlen[-1] / float(self.nnodes-1))

            # update node positions
            path_new = [self.path[0]]
            for i, _ in enumerate(self.path[1:-1], start=1):
                k = i

                # TODO: this is how the original implementation does it, but it seems unstable?
                #while not ((sumlen[k] < sfrac[i+1]) and (sumlen[k+1] >= sfrac[i+1])):
                #    k += 1
                #    if i >= self.nnodes:
                #        raise ValueError(" >>> ERROR: Reparametrization of path failed!")

                dr = sfrac[i] - sumlen[k] 
                vec = self.path[k+1] - self.path[k]
                vec /= torch.linalg.norm(vec)
                path_new.append(self.path[k] + dr * vec)

            path_new.append(self.path[-1])
            self.path = path_new.copy()

            # get actual length of path
            l = [0]
            sumlen = [0]
            for i, coords in enumerate(self.path[1:], start=1):
                l.append(torch.linalg.norm(coords - self.path[i-1]))
                sumlen.append(sumlen[-1] + l[-1])
            iter += 1

        self.boundary_nodes = self._get_boundary(self.path)
        
        if self.verbose:
            crit = abs(sumlen[-1]-prevsum)
            if crit < tol:
                print(f" >>> INFO: Reparametrization of Path converged in {iter} steps. Max(delta d_ij)={crit:.3f}.")
            else:
                print(f" >>> WARNING: Reparametrization of Path not converged in {max_step} steps. Max(delta d_ij)={crit:.3f}.")        

    @staticmethod
    def _smooth_string(path: list, s: float=0.0):
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
        """Calculates distance according to chosen metric

        Args:
            z: reduced coords 

        Return:
            rmsds: list of distances of z to every node of path
        """
        d = []
        for i in range(self.nnodes):
            d.append(self.get_distance(z, self.path[i], metric=self.metric))
            self.path[i] = self.path[i].detach()
        return d

    @staticmethod
    def get_distance(
        coords: torch.tensor, 
        reference: torch.tensor,
        metric: str="RMSD"    
    ) -> torch.tensor:
        """Get distance between coordinates and reference calculated by `metric`
        
        Available metrics are: 
            `RMSD`: Root mean square deviation
            `MSD`: Mean square deviation
            `kabsch`: Root mean square deviation of optimally fitted coords
            `KMSD`: Mean square deviation of optimally fitted coords
            `distance`: Absolute distance 

        Args:
            coords: coordinates 
            reference: coordinates of reference
            metric: distance metric

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
                " >>> INFO: Available distance metrics in PathCV: `RMSD`, `MSD`, `kabsch`, `KMSD`, `distance`"
            )

        return d.type(torch.float64)

    def _project_coords_on_line(self, z: torch.tensor, q: list) -> torch.tensor:
        """Projection of coords on line

        Args:
            z: reduced coords 
            q: coords of two nodes that span line on which to project

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

    def _get_closest_nodes(self, z: torch.tensor, dists: list) -> tuple:
        """get two closest nodes of path

        Args:
            z: reduced coordinates
            rmsds: list of rmsds of z to path nodes

        Returns:
            closest_index: list with indices of two closest nodes to z  
            closest_coords: list with coordinates of two closest nodes to z
        """
        dists.insert(0, self.get_distance(z, self.boundary_nodes[0], metric=self.metric))
        dists.append(self.get_distance(z, self.boundary_nodes[1], metric=self.metric))
        
        distmin1, idxmin1 = 99999, 0
        distmin2, idxmin2 = 99999, 1
        for i, dist in enumerate(dists):
            if dist < distmin1:
                distmin2 = distmin1
                idxmin2 = idxmin1
                distmin1 = dist
                idxmin1 = i
            elif (dist < distmin2):
                distmin2 = dist
                idxmin2 = i

        self.path.insert(0, self.boundary_nodes[0])
        self.path.append(self.boundary_nodes[1])
        closest_coords = [self.path[idxmin1], self.path[idxmin2]]
        del self.path[0]
        del self.path[-1]

        if self.verbose:
            # if path gets highly irregualar this can fail, so print warning
            if abs(idxmin1 - idxmin2) != 1:
                print(
                    f" >>> WARNING: Two closest nodes of PathCV ({idxmin1-1,idxmin2-1}) are not neighbours!"
                ) 
        
        return [idxmin1-1, idxmin2-1], closest_coords

    def _interpolate(self, n_interpolate):
        """Add nodes by linear interpolation or remove nodes by slicing
        
        Args:
            n_interpolate: Number of interpolated nodes between two original nodes
        """
        if n_interpolate > 0:
            # fill path with n_interplate nodes per pair
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

        elif n_interpolate < 0:
            # slice path to keep only every n_interpolate node 
            self.path = self.path[::abs(n_interpolate)]
        
        self.nnodes = len(self.path)

    def _reduce_path(self):
        """reduce path nodes to only active coordinates
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
                dcd.write(BOHR_to_ANGSTROM * coords.view(int(torch.numel(coords)/3),3).detach().numpy())



