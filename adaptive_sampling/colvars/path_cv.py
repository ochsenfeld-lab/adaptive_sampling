import time
import math
import torch
from typing import List, Union

from .utils import *

class PathCV:
    """Adaptive Path Collective Variables

    Args:
        guess_path: filename of `.xyz` or `.npy` with initial path 
        active: list of atom indices included in PathCV
            if `coordinate_system=cv_space`, has to contain cv definitions to define CV space 
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
        reduce_path: if Cartesian input path should be tranlated into CV space
        adaptive: if adaptive, path converges to average CV density perpendicular to path
        update_interval: number of steps between updates of adaptive path
        half_life: number of steps til weight of original path is half due to updates
        walkers: list of paths to `local_path_file` of multiple walkers contributing to path updates (if None: single walker)
        local_path_file: filename to dump path information of local simulation to share path between multiple walkers
        requires_z: if distance to path should be calculated and stored for confinement to path
        device: desired device of torch tensors 
        ndim: number of dimensions (2 for 2D test potentials, 3 else)
        verbose: if verbose information should be printed
    """
    def __init__(
        self, 
        guess_path: str=None, 
        active: List[Union[int, list]]=None, 
        n_interpolate: int=0,
        smooth_damping: float=0.0,
        reparam_steps: int=1,
        reparam_tol: float=0.01,
        coordinate_system: str="Cartesian",
        metric: str="RMSD",
        reduce_path: bool=True,
        adaptive: bool=False,
        update_interval: int=100,
        half_life: float=-1, 
        walkers: List[str]=None,
        local_path_file: str="local_path_data",
        requires_z: bool=False,
        device: str='cpu',
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
        self.walkers = walkers
        self.local_path_file = local_path_file
        self.requires_z = requires_z
        self.device = device
        self.ndim = ndim
        self.verbose = verbose
        self.closest_prev = None

        # initialize path
        self.path, self.nnodes = read_path(
            self.guess_path, 
            ndim=self.ndim,
        )

        # translates cartesian input path into CV space 
        if reduce_path:
            self._reduce_path()
        self._interpolate(n_interpolate)

        # accumulators for path update
        if self.adaptive:
            self.n_updates = -4 # TODO: don't count calls during init
            self.weights = torch.zeros(self.nnodes, device=self.device, requires_grad=False)    
            self.avg_dists = [torch.zeros_like(self.path[0], device=self.device, requires_grad=False) for _ in range(self.nnodes)]

            if self.walkers != None:
                self._dump_update_data()
                self.path = self._shared_path()

            if half_life < 0:
                self.fade_factor = 1.0
            else:
                self.fade_factor = math.exp(-math.log(2.) / float(self.half_life))

        # smooth path and ensure quidistant nodes
        self.path = PathCV.reparametrize_path(
            self.path, 
            smooth_damping=self.smooth_damping, 
            tol=self.reparam_tol, 
            max_step=self.reparam_steps, 
            verbose=self.verbose,
        )
        self.boundary_nodes = self._get_boundary(self.path) # trailing nodes

        if self.verbose:
            print(f" >>> INFO: Initialization of PathCV with {self.nnodes} nodes finished.")

    def calculate_path(self, coords: torch.tensor) -> torch.tensor:
        """Calculates arithmetic PathCV according to
        Branduardi, et al., J. Chem. Phys. (2007): https://doi.org/10.1063/1.2432340

        Args:
            coords: cartesian coordinates 

        Returns:
            path_cv: progress anlong path in range [0,1]
        """
        if self.path[0].shape == coords.shape:
            z = torch.clone(coords).to(self.device, non_blocking=True)
        else:
            z = convert_coordinate_system(
                coords, self.active, coord_system=self.coordinates, ndim=self.ndim
            ).to(self.device, non_blocking=True)

        rmsds = torch.stack(
            [self.get_distance(z, self.path[i], metric=self.metric) for i in range(self.nnodes)]
        )
        
        la = 1. / torch.min(rmsds)
        sm = torch.softmax(-la * rmsds, dim=0)
        self.path_cv = 1.0 / (self.nnodes - 1) * torch.sum(torch.arange(self.nnodes) * sm)

        # distance from path
        if self.requires_z:
            self.path_z = -1. / la * torch.logsumexp(-la * rmsds, dim=0)
            self.grad_z = torch.autograd.grad(
                self.path_z, coords, allow_unused=True, retain_graph=True,
            )[0]
            self.grad_z = self.grad_z.detach().numpy()
            
        if self.adaptive:
            raise NotImplementedError(f" >>> ERROR: Adaptive path updates only implemented for gpath.")
            #min_idx = self._get_closest_nodes(z, rmsds.tolist())
            #min_coords = []
            #for idx in min_idx:
            #    if idx == -1:
            #        min_coords.append(self.boundary_nodes[0])
            #    elif idx == self.nnodes:
            #        min_coords.append(self.boundary_nodes[1])
            #    else:
            #        min_coords.append(self.path[idx])
            #
            #self.update_path(z, q=min_coords)

        return self.path_cv

    def calculate_gpath(self, coords: torch.tensor) -> torch.tensor:
        """Calculate geometric PathCV according to
        Leines et al., Phys. Ref. Lett. (2012): https://doi.org/10.1103/PhysRevLett.109.020601
        
        Args:
            coords: cartesian coordinates

        Returns:
            path_cv: progress anlong path in range [0,1]
        """
        if self.path[0].shape == coords.shape:
            z = torch.clone(coords).to(self.device, non_blocking=True)
        else:
            z = convert_coordinate_system(
                coords, self.active, coord_system=self.coordinates, ndim=self.ndim
            ).to(self.device, non_blocking=True)
       
        dists = self._get_distance_to_path(z)
        idx_nodemin = self._get_closest_nodes(z, dists, regulize=False)
        
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

            self.n_updates += 1

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
            self.n_updates += 1
            s = self._project_coords_on_line(z, q)
            dist_ij = self.get_distance(self.path[0], self.path[1], metric=self.metric)
            for j, _ in enumerate(self.path[1:-1], start=1):
                w = max([0, 1 - self.get_distance(self.path[j], s, metric=self.metric) / dist_ij])
                self.avg_dists[j] += w * (z-s)
                self.weights[j] *= self.fade_factor
                self.weights[j] += w

        # don't count calls during init
        if self.n_updates < 0:
            self.weights = torch.zeros_like(self.weights, device=self.device, requires_grad=False)
            self.avg_dists = [torch.zeros_like(self.avg_dists[0], device=self.device, requires_grad=False) for _ in range(self.nnodes)]
       
        # avoid performance drop due to buildup of large torch graphs in long simulations
        self.weights = self.weights.detach()
        self.avg_dists = [avg_dist.detach() for avg_dist in self.avg_dists]

        # update path from accumulators all `self.update_interval` steps
        if self.n_updates == self.update_interval:
            
            for j in range(self.nnodes-2):
                if self.weights[j+1] > 0:
                    self.path[j+1] += (self.avg_dists[j+1] / self.weights[j+1])
            
            # calculate weighted path average from multiple walkers listed in `self.walkers`
            if self.walkers != None:
                self._dump_update_data()
                self.path = self._shared_path()

            # smooth path and ensure equidistant nodes 
            self.path = PathCV.reparametrize_path(
                self.path, 
                smooth_damping=self.smooth_damping, 
                tol=self.reparam_tol, 
                max_step=self.reparam_steps,
                verbose=self.verbose,
            )

            # trailing nodes
            self.boundary_nodes = self._get_boundary(self.path) 
            
            # reset accumulators
            self.n_updates = 0 
            self.avg_dists = [torch.zeros_like(self.avg_dists[0], device=self.device, requires_grad=False) for _ in range(self.nnodes)]

    @staticmethod
    def reparametrize_path(
        path: torch.tensor, 
        tol: float=0.01, 
        max_step: int=10,
        smooth_damping: float=0.0,
        verbose: bool=True,
    ) -> list:
        """Reparametrization of path to ensure equidistant nodes
        see: Maragliano et al., J. Chem. Phys. (2006): https://doi.org/10.1063/1.2212942
        
        Args:
            path: list of coordinates of path nodes
            smooth_damping: 
            tol: tolerance for convergence, difference of absolute path length to previous cycle
            max_step: maximum number of iterations
            smooth_damping: damping factor for smoothing in range(0,1)
            verbose: print info massage
        
        Returns:
            path: reparametrized list of coordinates of path nodes
        """ 
        if smooth_damping > 0:
            path = PathCV.smooth_string(path, s=smooth_damping)

        nnodes = len(path)

        # get length of path
        L, sumlen = [0], [0]
        for i, coords in enumerate(path[1:], start=1):
            L.append(torch.linalg.norm(coords - path[i-1]))
            sumlen.append(sumlen[-1] + L[-1])

        prevsum, iter = 0, 0
        while abs(sumlen[-1]-prevsum) > tol and iter <= max_step:
            prevsum = sumlen[-1]

            # cumulative target distance between nodes
            sfrac = []
            for i in range(nnodes):
                sfrac.append(i * sumlen[-1] / float(nnodes-1))

            # update node positions
            path_new = [path[0]]
            for i, _ in enumerate(path[1:-1], start=1):
                k = 0
                while not ((sumlen[k] < sfrac[i]) and (sumlen[k+1] >= sfrac[i])):
                    k += 1
                    if i >= nnodes:
                        raise ValueError(" >>> ERROR: Reparametrization of path failed!")
                dr = sfrac[i] - sumlen[k] 
                vec = path[k+1] - path[k]
                vec /= torch.linalg.norm(vec)
                path_new.append(path[k] + dr * vec)
            path_new.append(path[-1])
            path = path_new.copy()

            # get new length of path
            L, sumlen = [0], [0]
            for i, coords in enumerate(path[1:], start=1):
                L.append(torch.linalg.norm(coords - path[i-1]))
                sumlen.append(sumlen[-1] + L[-1])
            iter += 1
        
        if verbose:
            crit = abs(sumlen[-1]-prevsum)
            if crit < tol:
                print(f" >>> INFO: Reparametrization of Path converged in {iter} steps. Final convergence: {crit:.6f}.")
            else:
                print(f" >>> WARNING: Reparametrization of Path not converged in {max_step} steps. Final convergence: {crit:.6f}.")

        return path    

    @staticmethod
    def smooth_string(path: list, s: float=0.0) -> list:
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

    def _calc_1overlambda(self):
        """get lambda parameter for smoothing of arithmetic path"""
        sumlen = 0
        for i, coords in enumerate(self.path[1:], start=1):
            d = self.get_distance(coords, self.path[i-1], metric=self.metric)
            sumlen += d
        return sumlen / (self.nnodes-1)
        
    def _get_distance_to_path(self, z: torch.tensor) -> list:
        """Calculates distance according to chosen metric

        Args:
            z: reduced coords 

        Return:
            rmsds: list of distances of z to every node of path
        """
        d = []
        for i in range(self.nnodes):
            self.path[i] = self.path[i].detach().to(self.device)
            d.append(self.get_distance(z, self.path[i], metric=self.metric))
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
            regulize: ensure that closest node moves no more than 1 index between steps
            
        Returns:
            min_idx: list with indices of two closest nodes to z  
        """
        dists.insert(0, self.get_distance(z, self.boundary_nodes[0], metric=self.metric))
        dists.append(self.get_distance(z, self.boundary_nodes[1], metric=self.metric))
        
        min1 = min2 = float('inf')
        min1_idx = min2_idx = -1
        for i, num in enumerate(dists):
            if num < min1:
                min2 = min1
                min2_idx = min1_idx
                min1 = num
                min1_idx = i
            elif num < min2:
                min2 = num
                min2_idx = i

        if regulize: 
            min1_idx, min2_idx = self._check_nodemin_prev(min1_idx, min2_idx, dists)
        
        if self.verbose:
            if abs(min1_idx - min2_idx) != 1:
                print(
                    f" >>> WARNING: Shortcutting path at node indices ({min1_idx-1, min2_idx-1})"
                )

        return [min1_idx-1, min2_idx-1]
        
    def _check_nodemin_prev(self, idx0: int, idx1: int, dists: list) -> int:
        """ensures that index of closest node changes no more than 1 to previous step
        """
        if self.closest_prev != None:
            if abs(self.closest_prev - idx0) > 1:
                if self.closest_prev >= self.nnodes+1:
                    d = [dists[self.closest_prev-1], dists[self.closest_prev]]
                if self.closest_prev == 0:
                    d = [dists[self.closest_prev], dists[self.closest_prev+1]]
                else:
                    d = [dists[self.closest_prev-1], dists[self.closest_prev], dists[self.closest_prev+1]]
                
                d.sort()
                idx0 = dists.index(d[0])
                idx1 = dists.index(d[1])
                
        self.closest_prev = idx0
        return idx0, idx1

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

    def _shared_path(self) -> List[torch.tensor]:
        """Synchronizes path with multiple walkers by calculating weighted average of path nodes for walkers listed in `self.walkers`

        Returns:
            global_path: average of path nodes from all walkers
        """
        if self.walkers==None:
            raise ValueError(" >>> ERROR: No other walkers specified for shared path!")

        global_path = [torch.zeros_like(node) for node in self.path]
        global_weights  = torch.zeros_like(self.weights)
        
        n_success = 0
        for i, walker in enumerate(self.walkers):
            
            if walker[-1] == '\n':
                walker = walker[:-1]

            try:
                data = torch.load(walker)
            except:
                if self.verbose:
                    print(f" >>> WARNING: Failed to read {i}th walker of shared path update!")
                continue

            n_success += 1
            for i, node in enumerate(data['path']):
                global_path[i] += node * data['weights'][i]
            global_weights += data['weights']

        for j, w in enumerate(global_weights):
            if w > 0:
                global_path[j] /= w
            else:
                global_path[j] = torch.clone(self.path[j])

        if self.verbose:
            print(f" >>> INFO: Synchronized path with {n_success} walkers.")

        return global_path

    def _dump_update_data(self):
            """Dumps data that is necessary to sync path with other walkers to `self.local_path_file`
            """
            if self.local_path_file[-4:] != '.pth':
                self.local_path_file += '.pth'

            for i, file in enumerate([self.local_path_file for _ in range(10)]):
                try:
                    torch.save(
                        {
                            "weights": self.weights,
                            "path": self.path
                        },
                        file,
                    ) 
                    break
                except:
                    # catches errors if other walker accesses file at the same time
                    if i == 9 and self.verbose:
                        print(f" >>> WARNING: Failed to dump multiple walker path data in `{file}`!")
                        break
                    time.sleep(0.1)
                    continue

    def write_update_info(self, filename: str="update_info"):
        """write accumulators for path update to `.npz` file

        Arg:
            filename
        """
        if self.adaptive:
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
