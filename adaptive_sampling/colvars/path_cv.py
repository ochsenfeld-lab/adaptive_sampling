import math
import torch

from .utils import *

class PathCV:
    """Adaptive Path Collective Variables

    Args:
        guess_path: filename of `.xyz` or `.npy` with initial path 
        active: list of atom indices included in PathCV
            if `coordinate_system=cv_space`, has to contain list of lists of indices for distances, angles and torsions that define CV space 
        n_interpolate: Number of nodes that are added between original nodes by linear interpolation,
            if negative, slice path nodes according to `self.path[::abs(n_interpolate)]`
        smooth_damping: Controls smoothing of path (0: no smoothing, 1: linear interpolation between neighbours)
        reparam_steps: Maximum number of cycles for reparamerization of path to ensure equidistant nodes
        reparam_tol: Tolerance for reparametrization of path to ensure equidistant nodes
        coordinate_system: coordinate system for calculation of PathCV
            available: `cartesian`, `zmatrix` or `cv_space`
        metric: Metric for calculation of distance of points
            `RMSD`: Root mean square deviation
            `MSD`: Mean square deviation
            `Kabsch`: Root mean square deviation of optimally fitted coords
            `KMSD`: Mean square deviation of optimally fitted coords
            `distance`: Absolute distance 
        adaptive: if adaptive, path converges to average CV density perpendicular to path
        update_interval: number of steps between updates of adaptive path
        half_life: number of steps til weight of original path is half due to updates
        requires_z: if distance to path should be calculated and stored for confinement to path
        ndim: number of dimensions (2 for 2D test potentials, 3 else)
        verbose: if verbose information should be printed
    """
    def __init__(
        self, 
        guess_path: str=None, 
        active: list=None, 
        n_interpolate: int=0,
        smooth_damping: float=0.0,
        reparam_steps: int=1,
        reparam_tol: float=0.01,
        coordinate_system: str="Cartesian",
        metric: str="RMSD",
        adaptive: bool=False,
        update_interval: int=100,
        half_life: float=-1, 
        requires_z: bool=False,
        ndim: int=3,
        verbose: bool=False,
    ):  
        if guess_path == None:
            raise ValueError(" >>> ERROR: You have to provide a guess path to the PathCV")
        self.guess_path = guess_path
        self.active = active 
        self.smooth_damping = smooth_damping
        self.reparam_steps = reparam_steps
        self.reparam_tol = reparam_tol
        self.coordinates = coordinate_system
        self.metric = metric
        self.adaptive = adaptive
        self.update_interval = update_interval
        self.half_life = half_life
        self.requires_z = requires_z
        self.ndim = ndim
        self.verbose = verbose
        
        # initialize path
        self.path, self.nnodes = read_path(
            self.guess_path, 
            ndim=self.ndim,
        )

        # if `.xyz` is given convert coordinate system 
        if self.guess_path[-3:] == 'xyz':
            self._reduce_path()

        self._interpolate(n_interpolate)
        self._reparametrize_path(tol=self.reparam_tol, max_step=self.reparam_steps)
        self.boundary_nodes = self._get_boundary(self.path)
        self.closest_prev = None

        # accumulators for path update
        if self.adaptive:
            self.n_updates = -4 # TODO: don't count calls during init
            self.weights = torch.zeros(self.nnodes)    
            self.avg_dists = [torch.zeros_like(self.path[0]) for _ in range(self.nnodes)]
            if half_life < 0:
                self.fade_factor = 1.0
            else:
                self.fade_factor = math.exp(-math.log(2.) / float(self.half_life))

        if self.verbose:
            print(f" >>> INFO: Initialization of PathCV with {self.nnodes} nodes finished.")

    def calculate_path(self, coords: torch.tensor) -> torch.tensor:
        """calculate PathCV according to
        Branduardi, et al., J. Chem. Phys. (2007): https://doi.org/10.1063/1.2432340

        Args:
            coords: cartesian coordinates 

        Returns:
            path_cv: progress anlong path in range [0,1]
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

        # position on path in range [0,1]
        self.path_cv = (1. / (self.nnodes-1)) * (term1 / term2)
        
        # distance from path
        if self.requires_z:
            self.path_z = -1. / la * torch.log(term2)
            self.grad_z = torch.autograd.grad(
                self.path_z, coords, allow_unused=True, retain_graph=True,
            )[0]
            self.grad_z = self.grad_z.detach().numpy()

        if self.adaptive:
            _, coords_nearest = self._get_closest_nodes(z, rmsds)
            self.update_path(z, q=coords_nearest)

        return self.path_cv

    def calculate_gpath(self, coords: torch.tensor) -> torch.tensor:
        """Calculate geometric PathCV according to
        Leines et al., Phys. Ref. Lett. (2012): https://doi.org/10.1103/PhysRevLett.109.020601
        
        Args:
            coords: cartesian coordinates

        Returns:
            path_cv: progress anlong path in range [0,1]
        """
        z = convert_coordinate_system(
            coords, self.active, coord_system=self.coordinates, ndim=self.ndim
        )
        dists = self._get_distance_to_path(z)
        idx_nodemin, coords_nodemin = self._get_closest_nodes(z, dists, regulize=True)
        
        # add boundary nodes to `self.path`
        self.path.insert(0, self.boundary_nodes[0])
        self.path.append(self.boundary_nodes[1])
        idx_nodemin = [i+1 for i in idx_nodemin]

        # start of main computation of PathCV   
        isign = idx_nodemin[0] - idx_nodemin[1]
        if isign > 1:
            isign = 1
        elif isign < 1:
            isign = -1

        idx_nodemin[1] = idx_nodemin[0] - isign
        idx_nodemin.append(idx_nodemin[0] + isign)
        
        v1 = self.path[idx_nodemin[0]] - z
        v3 = z - self.path[idx_nodemin[1]] 
        
        # if idx_nodemin[2] is out of bounds, use idx_nodemin[1]
        if idx_nodemin[2] < 0 or idx_nodemin[2] > self.nnodes:
            v2 = self.path[idx_nodemin[0]] - self.path[idx_nodemin[1]]
        else:
            v2 = self.path[idx_nodemin[2]] - self.path[idx_nodemin[0]]

        # get `s` component of PathCV (progress along path)
        v1 = v1.view(torch.numel(v1))
        v2 = v2.view(torch.numel(v2))
        v3 = v3.view(torch.numel(v3))

        v1v1 = torch.matmul(v1, v1)
        v1v2 = torch.matmul(v1, v2)
        v2v2 = torch.matmul(v2, v2) 
        v3v3 = torch.matmul(v3, v3)

        root = torch.sqrt(torch.abs(torch.square(v1v2) - v2v2 * (v1v1 - v3v3)))
        dx = 0.5 * ((root - v1v2) / v2v2 - 1.)
        self.path_cv = ((idx_nodemin[0]-1) + isign * dx) / (self.nnodes-1)

        if self.requires_z or self.adaptive:
            v = z - self.path[idx_nodemin[0]] - dx * (
                self.path[idx_nodemin[0]] - self.path[idx_nodemin[1]]
            )

        # finished with PathCV, remove boundary nodes from `self.path`
        del self.path[0]
        del self.path[-1]
        idx_nodemin = [idx-1 for idx in idx_nodemin]

        # get perpendicular `z` component (distance to path)
        if self.requires_z: 
            self.path_z = torch.linalg.norm(v)
            self.grad_z = torch.autograd.grad(
                self.path_z, coords, allow_unused=True, retain_graph=True,
            )[0]
            self.grad_z = self.grad_z.detach().numpy()

        # accumulate average distance from path for path update
        if self.adaptive:

            w2 = -1. * dx
            w1 = 1. + dx
            if w1 > 1:
                w1, w2 = 1.0, 0.0 
            elif w2 > 1:
                w1, w2 = 0.0, 1.0
            
            if idx_nodemin[0] > 0 and idx_nodemin[0] < self.nnodes-1:
                self.avg_dists[idx_nodemin[0]] += w1 * v
                self.weights[idx_nodemin[0]]   *= self.fade_factor 
                self.weights[idx_nodemin[0]]   += w1 
            
            if idx_nodemin[1] > 0 and idx_nodemin[1] < self.nnodes-1:
                self.avg_dists[idx_nodemin[1]] += w2 * v
                self.weights[idx_nodemin[1]]   *= self.fade_factor 
                self.weights[idx_nodemin[1]]   += w2
            
            self.update_path(z, q=None)
        
        return self.path_cv

    def update_path(self, z: torch.tensor, q: list=None):
        """Update path nodes to ensure smooth convergence to the MFEP
        
        see: Ortiz et al., J. Chem. Phys. (2018): https://doi.org/10.1063/1.5027392

        Args:
            z: reduced coords 
            q: coords of two neighbour nodes of z
        """
        if q != None:
            s = self._project_coords_on_line(z, q)
            dist_ij = self.get_distance(self.path[0], self.path[1], metric=self.metric)
            for j, _ in enumerate(self.path[1:-1], start=1):
                w = max([0, 1 - self.get_distance(self.path[j], s, metric=self.metric) / dist_ij])
                self.avg_dists[j] += w * (z-s)
                self.weights[j] *= self.fade_factor
                self.weights[j] += w

        # don't count calls during init
        if self.n_updates < 0:
            self.weights = torch.zeros_like(self.weights)
            self.avg_dists = [torch.zeros_like(self.avg_dists[0]) for _ in range(self.nnodes)]
       
        # update path all `self.update_interval` steps
        self.n_updates += 1
        if self.n_updates == self.update_interval:
            new_path = self.path.copy()
            for j in range(self.nnodes-2):
                if self.weights[j+1] > 0:
                    new_path[j+1] += (self.avg_dists[j+1] / self.weights[j+1])

            self.path = new_path.copy()
            self._reparametrize_path(tol=self.reparam_tol, max_step=self.reparam_steps)

            # reset accumulators
            self.n_updates = 0 
            self.avg_dists = [torch.zeros_like(self.avg_dists[0]) for _ in range(self.nnodes)]

    def _reparametrize_path(
        self, 
        tol: float=0.01, 
        max_step: int=10,
        smooth: bool=True,
    ):
        """Reparametrization of path to ensure equidistant nodes
        see: Maragliano et al., J. Chem. Phys. (2006): https://doi.org/10.1063/1.2212942
        
        Args:
            tol: tolerance for convergence, difference of absolute path length to previous cycle
            max_step: maximum number of iterations
            smooth: if path should be smoothed to remove kinks
        """ 
        # TODO: When smoothing only once there might be too many kinks in the path in some cases?
        if smooth:
            self.path = self._smooth_string(self.path, s=self.smooth_damping)

        # get length of path
        L, sumlen = [0], [0]
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

            # TODO: Smoothing in every cycle might introduce too much corner cutting?
            #if smooth:
            #    self.path = self._smooth_string(self.path, s=self.smooth_damping)

            # update node positions
            path_new = [self.path[0]]
            for i, _ in enumerate(self.path[1:-1], start=1):
                k = 0
                while not ((sumlen[k] < sfrac[i]) and (sumlen[k+1] >= sfrac[i])):
                    k += 1
                    if i >= self.nnodes:
                        raise ValueError(" >>> ERROR: Reparametrization of path failed!")
                dr = sfrac[i] - sumlen[k] 
                vec = self.path[k+1] - self.path[k]
                vec /= torch.linalg.norm(vec)
                path_new.append(self.path[k] + dr * vec)
            path_new.append(self.path[-1])
            self.path = path_new.copy()

            # get new length of path
            L, sumlen = [0], [0]
            for i, coords in enumerate(self.path[1:], start=1):
                L.append(torch.linalg.norm(coords - self.path[i-1]))
                sumlen.append(sumlen[-1] + L[-1])
            iter += 1

        self.boundary_nodes = self._get_boundary(self.path)
        
        if self.verbose:
            crit = abs(sumlen[-1]-prevsum)
            if crit < tol:
                print(f" >>> INFO: Reparametrization of Path converged in {iter} steps. Final convergence: {crit:.6f}.")
            else:
                print(f" >>> WARNING: Reparametrization of Path not converged in {max_step} steps. Final convergence: {crit:.6f}.")        

    @staticmethod
    def _smooth_string(path: list, s: float=0.0) -> list:
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
    def _get_boundary(path: list) -> list:
        """compute one final lower and upper node by linear interpolation of path
        
        Args:
            path: list of path nodes
        
        Returns:
            boundaries: list with lower and upper boundary node
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
            d.append(float(self.get_distance(z, self.path[i], metric=self.metric)))
            self.path[i] = self.path[i].detach()  # keep path detached for performance reasons
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
        shape_z = z.shape        # to reshape results
        ncoords = torch.numel(z) # to flatten inputs

        z = z.view(ncoords)
        q0 = q[0].view(ncoords)
        q1 = q[1].view(ncoords)
        ap = z - q0
        ab = q1 - q0

        return (q0 + torch.matmul(ap,ab) / torch.matmul(ab,ab) * ab).view(shape_z)

    def _get_closest_nodes(self, z: torch.tensor, dists: list, regulize: bool=False) -> tuple:
        """get two closest nodes of path

        Args:
            z: reduced coordinates
            dists: list of distances of z to path nodes

        Returns:
            closest_index: list with indices of two closest nodes to z  
            closest_coords: list with coordinates of two closest nodes to z
        """
        dists.insert(0, float(self.get_distance(z, self.boundary_nodes[0], metric=self.metric)))
        dists.append(float(self.get_distance(z, self.boundary_nodes[1], metric=self.metric)))
        
        dists_sorted = dists.copy()
        dists_sorted.sort()

        idxmin1 = dists.index(dists_sorted[0])
        idxmin2 = dists.index(dists_sorted[1])

        if regulize: 
            idxmin1 = self._check_nodemin_prev(idxmin1, dists)

        self.path.insert(0, self.boundary_nodes[0])
        self.path.append(self.boundary_nodes[1])
        closest_coords = [self.path[idxmin1], self.path[idxmin2]]
        del self.path[0]
        del self.path[-1]
        
        if self.verbose:
            # if path gets highly irregualar, print warning
            if abs(idxmin1 - idxmin2) != 1:
                print(
                    f" >>> WARNING: Two closest nodes of path ({idxmin1-1,idxmin2-1}) are not neighbours!"
                )

        return [idxmin1-1, idxmin2-1], closest_coords
        
    def _check_nodemin_prev(self, idx: int, dists: list) -> int:
        """ensures that index of closest node changes no more than 1 to previous step
        """
        if self.closest_prev != None:
            if abs(self.closest_prev - idx) > 1:
                if self.closest_prev >= self.nnodes+1:
                    d = [dists[self.closest_prev-1], dists[self.closest_prev]]
                if self.closest_prev == 0:
                    d = [dists[self.closest_prev], dists[self.closest_prev+1]]
                else:
                    d = [dists[self.closest_prev-1], dists[self.closest_prev], dists[self.closest_prev+1]]
                
                d.sort()
                idx = dists.index(d[0])
                
        self.closest_prev = idx
        return idx

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
            print(f" >>> INFO: Reduced coordinate system to {torch.numel(self.path[0])} {self.coordinates} coordinates.")

    def write_update_info(self, filename: str="update_info"):
        """write accumulators for path update to `.npz` file

        Arg:
            filename
        """
        import numpy as np
        
        # convert to numpy
        w = self.weights.detach().numpy()
        avg_dist = []
        for dists in self.avg_dists:
            avg_dist.append(dists.detach().numpy())
        
        np.savez(
            file=filename,
            weights=w,
            avg_dists=avg_dist,
            n_updates=self.n_updates,
        )

    def write_path(self, filename: str="path_cv.npy"):
        """write nodes of PathCV to dcd trajectory file

        Args:
            filename: filename for path output, can be in `.npy` or `.dcd` format 
        """
        if filename[-3:] == "npy":
            import numpy as np
            path_tmp = []
            for coords in self.path:
                path_tmp.append(coords.detach().numpy())
            np.save(filename, path_tmp, allow_pickle=True)
        else:
            import mdtraj
            from ..units import BOHR_to_ANGSTROM

            dcd = mdtraj.formats.DCDTrajectoryFile(filename, 'w', force_overwrite=True)
            for coords in self.path:
                dcd.write(BOHR_to_ANGSTROM * coords.view(int(torch.numel(coords)/3),3).detach().numpy())
            dcd.close()
