import os
import numpy as np
from typing import List
from ..units import *

try:
    from ase import io
    from ase.neb import NEB
except ImportError:
    import sys as _sys
    print(">>> adaptive-sampling: Module 'ase' not found, will not import 'FENEB'", file=_sys.stderr)
    del _sys

class FENEB:
    """Free Energy NEB method for path optimisation in cartesian coordinates

    Based on: Semelak, Jonathan A., et al. JCTC (2023): <https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00366>

    Args:
        fep_k: force constant for umbrella integration, has to match force constant of confinement in MD simulations
        neb_k: force constant of spring force
        extremes: List of filenames of xyz file for reactand and product structure
        path: `.npy` file with NEB nodes, if None, a new path will be created by interpolation between extremes
        nimages: number of NEB images without extremes
        conf_spring: convergence criterion for uncouled band optimization
        maxiter_spring: maximum iterations for uncoupled band optimization
        active: Indices of active atoms that build the cartesian reaction coordinate space (RCS)
        idpp: if True, use improved path interpolation with IDPP method
        load_from_dcd: if True, load MD data form DCD file, otherwise a simple `.dat` with RCS data in columns is assumed
        top: Topology filename (e.g. .pdb, .prmtop, .parm7), required to load DCD
        write_xyz: write path nodes to xyz files
        convert_units: convert Angstrom to Bohr for all corresponding inputs
        verbose: print verbose information
    """
    def __init__(
        self,
        fep_k: float,
        neb_k: float,
        extremes: List[str],
        path: str=None,
        nimages: int=10,
        conf_spring: float=0.1,
        maxiter_spring: int=1000,
        active: List[int]=None,
        idpp: bool=False,
        load_from_dcd: bool=True,
        top: str=None,
        write_xyz: bool=False,
        convert_units: bool=True,
        verbose: bool=False,
    ):
        self.convert_units = convert_units
        if convert_units:
            self.neb_k = neb_k * BOHR_to_ANGSTROM * BOHR_to_ANGSTROM  
            self.fep_k = fep_k * BOHR_to_ANGSTROM * BOHR_to_ANGSTROM 
        else:
            self.neb_k = neb_k
            self.fep_k = fep_k

        self.verbose = verbose
        self.active = active
        self.conf_spring = conf_spring
        self.maxiter_spring = maxiter_spring
        self.nimages = nimages

        self.load_from_dcd = load_from_dcd
        if self.load_from_dcd:
            self.top = top
            if self.top == None:
                print(' >>> FENEB: you need to give a topology (e.g. pdb or prmtop) for dcd import!')
            if self.verbose:
                print(f' >>> FENEB: Topology for dcd import: {self.top}')
        
        self.optstep = 0

        self.initial = io.read(extremes[0]) # always store one ase.Atoms object for io operations
        if self.active is None:
            self.active = [i for i in range(len(self.initial.get_atomic_numbers()))]
        self.natoms = len(self.initial.get_atomic_numbers())

        if path == None:
            self.final = io.read(extremes[1])
            self.path = self.make_path(idpp)
            if write_xyz:
                self.write_nodes_to_xyz()
        else:
            self.path = np.load(path)
            self.path = [node for node in self.path]

    def step(
        self, 
        feg: np.ndarray, 
        active_relaxed: list=None,
        uncoupled_band_opt: bool=True, 
        frozen_extremes: bool=True,
    ) -> dict:
        """ NEB optimisation step (Steepest Descent)

        Args:
            feg: Cartesian free energy gradient with shape (N_nodes, M_atoms, 3)
            active_relaxed: list of surrounding atom indices where full optimization is performed
            uncoupled_band_opt: full optimization of spring force
            frozen_extremes: if False, band extremes are fully optimized
        """
        self.optstep += 1
        confs_fes, scaling_factors = [], []
        for i, _ in enumerate(self.path):
            
            feg_i = np.asarray(feg[i])[self.active].flatten()
            
            if (0 < i < len(self.path)-1):
            
                # tanget vector to path
                tau = self.tangent_vector([
                    self.path[i-1][self.active],
                    self.path[i][self.active],
                    self.path[i+1][self.active],
                ])  

                # free energy gradient orthogonal to path
                neb_force = feg_i - np.dot(feg_i, tau) * tau
            
                if not uncoupled_band_opt:
                    # spring force
                    neb_force -= self.neb_k * (
                        np.linalg.norm(self.path[i+1][self.active]-self.path[i][self.active])-
                        np.linalg.norm(self.path[i][self.active]-self.path[i-1][self.active])
                    ) * tau

            else:
                neb_force = feg_i

            force = np.zeros_like(self.path[i])
            if active_relaxed is not None:
                force[active_relaxed] = np.asarray(feg[i])[active_relaxed]
            force[self.active] = neb_force.reshape((len(self.active),3))
            
            if (i==0 or i==len(self.path)-1) and frozen_extremes:
                force = np.zeros_like(self.path[i])

            force_norm_max, scaling_factor = self._get_scaling_factor(force, convert_units=self.convert_units)
            if force_norm_max > 0.0:
                self.path[i] -= scaling_factor * force / force_norm_max 
            else:
                if self.verbose and (i!=0 and i!=len(self.path)-1 and not frozen_extremes) :
                    print(' >>> WARNING: Maximum norm of NEB force is zero')

            confs_fes.append(force_norm_max)
            scaling_factors.append(scaling_factor)

        if uncoupled_band_opt:
            conf_spring, iter_spring = self.opt_node_spacing(outfreq=100, newton_raphson=False)
        else: 
            conf_spring, iter_spring = 0.0, 0.0

        if self.verbose:
            if uncoupled_band_opt:
                print(
                    f' >>> FENEB: Iter {self.optstep:3d}:\tMax(Forces) = {np.max(confs_fes):3.6f} (idx: {np.argmax(confs_fes)+1:3d}),\tMax(F_spring) = {np.max(conf_spring):3.6f} (Iters: {iter_spring:4d})'
                )
            else:
                print(
                    f' >>> FENEB: Iter {self.optstep:3d}:\tMax(Forces) = {np.max(confs_fes):3.6f} (idx: {np.argmax(confs_fes)+1:3d})'
                )

        results = {
            'Iter': self.optstep, 
            'Max_force': np.max(confs_fes), 
            'Max_force_idx': np.argmax(confs_fes)+1,
            'Max_force_string': conf_spring,
        }
        return results

    def opt_node_spacing(self, outfreq: int=100, newton_raphson=False):
        """ full optimization of spring force 
        
        Args:
            outfreq: output is writen every outfreq step
            newton_raphson: use newton_raphson algorithm instead of Steepest Descent (CAUTION: its bugy)
        """
        conf_spring = 9999
        iter_spring = -1

        if self.verbose:
            print(" >>> FENEB: Starting full optimization of spring force")

        while True:
            
            tmp_path = self.path.copy()
            confs_spring = []
            for i, _ in enumerate(self.path[1:-1]):
                
                tau = self.tangent_vector([
                    self.path[i][self.active],
                    self.path[i+1][self.active],
                    self.path[i+2][self.active],
                ], newton_raphson=newton_raphson)      
                
                n_diff1 = np.linalg.norm(self.path[i+2][self.active]-self.path[i+1][self.active])
                n_diff2 = np.linalg.norm(self.path[i+1][self.active]-self.path[i+0][self.active])
                fspring = self.neb_k * (n_diff1 - n_diff2) * tau 

                conf, spring_scaling = self._get_scaling_factor(fspring, convert_units=self.convert_units)
                confs_spring.append(conf)
                
                if conf > 0.0:
                    if newton_raphson:
                        v = self.path[i+1][self.active] / n_diff1 + self.path[i+1][self.active] / n_diff2
                        hessian = self.neb_k * (np.outer(v.flatten(), tau.flatten()))
                        delta_R = np.matmul(np.linalg.pinv(hessian), fspring.T)
                    else: 
                        delta_R = spring_scaling * fspring / conf

                    # move path node i+1
                    tmp_path[i+1][self.active] += delta_R.reshape((len(self.active),3))

                else:
                    if self.verbose:
                        print(' >>> WARNING: Norm of FENEB force is zero')

            iter_spring += 1
            conf_spring = np.max(confs_spring)
            self.path = tmp_path.copy()

            if iter_spring%outfreq == 0:
                if self.verbose:
                    print(f" >>> Iter: {iter_spring:5d}, Max(F_spring): {conf_spring:14.6f} (node index: {np.argmax(confs_spring)})")
            
            if conf_spring < self.conf_spring:
                if self.verbose:
                    print(f" >>> FENEB: Spring optimization converged after {iter_spring} iterations.")
                break
            
            if self.maxiter_spring <= iter_spring:
                if self.verbose:
                    print(f" >>> FENEB: Final spring force after {iter_spring} steps: {conf_spring}.")
                break

        return conf_spring, iter_spring

    @staticmethod
    def tangent_vector(image, newton_raphson: bool=False):
        """ tanget vector to path """
        if newton_raphson:
            # avoids dependance of hessian on gradient of tau
            tau = image[2] - image[0]
            return tau / np.linalg.norm(tau)
        
        x_prev = image[1] - image[0]
        x_prev /= np.linalg.norm(x_prev)
        x_next = image[2] - image[1]
        x_next /= np.linalg.norm(x_next)
        tau = x_prev + x_next
        tau /= np.linalg.norm(tau)    
        return tau.flatten()       

    @staticmethod
    def _gradient_tangent(image):
        """ Gradient of tangent vector (tau) """
        diff1 = image[2] - image[1]
        diff2 = image[0] - image[1]
        n_diff1 = np.linalg.norm(diff1)
        n_diff2 = np.linalg.norm(diff2)
        vec1 = diff1 / n_diff1
        vec2 = diff2 / n_diff2
        d1 = np.divide(vec1, diff1, out=np.zeros_like(vec1), where=(diff1 != 0))
        d2 = np.divide(vec2, diff2, out=np.zeros_like(vec2), where=(diff2 != 0))
        grad = vec1 + vec2 
        grad *= d1 + d2 - 1. / n_diff1 - 1. / n_diff2
        grad /= np.linalg.norm(vec1 + vec2)
        return grad

    @staticmethod
    def _get_scaling_factor(forces, convert_units: bool=True):
        """ Get SD scaling based on heuristic criterion
        """
        atom_fnorm_max = np.max(np.linalg.norm(forces.reshape((int(len(forces.flatten())/3),1,3)), axis=2))
        
        if convert_units:
            unit_conversion = BOHR_to_ANGSTROM
        else: 
            unit_conversion = 1.0                            

        if atom_fnorm_max > 5.0:
            scaling_factor = 0.005 / unit_conversion
        elif atom_fnorm_max > 3.0:
            scaling_factor = 0.002 / unit_conversion
        else:
            scaling_factor = 0.0001 / unit_conversion

        return atom_fnorm_max, scaling_factor

    def make_path(self, idpp: bool) -> List[np.ndarray]:
        """Build initial path by linear interpolation between endpoints

        Args:
            endpoints: list of ase molecules that represent endpoints of path
            idpp: if True, get improved path guess from IDPP method
        """
        images = [self.initial]
        images += [self.initial.copy() for _ in range(self.nimages)]
        images += [self.final]

        if len(self.active) != len(images[0].get_positions()):
            
            if self.verbose:
                print(" >>> FENEB: Fixing atom positions that are not `active` in path interpolation")
            
            apply_constraint = True
            from ase.constraints import FixAtoms
            c = FixAtoms(indices=np.delete(np.arange(len(images[0].get_positions())), self.active))
            for img in images:
                img.set_constraint(c)
                
        else:
            apply_constraint = False

        neb = NEB(images)
        if idpp:
            neb.interpolate('idpp', apply_constraint=apply_constraint)
        else:
            neb.interpolate(apply_constraint=apply_constraint)

        if self.convert_units:
            return [image.get_positions() / BOHR_to_ANGSTROM for image in images]
        else:
            return [image.get_positions() for image in images]

    def calc_feg(self, startframe: int=0, move_nodes_to_mean: bool=True, active_relaxed: list=None, dim: int=3) -> np.ndarray:
        """Get free energy gradient for Umbrella Integration

        Args:
            move_nodes_to_mean: set nodes to mean coords of Umbrella Window
            dim: number of dimensions, set to 2 for 2D test potentials
        """
        mean_coords = np.zeros_like(self.path)
        self.feg = np.zeros_like(self.path)
        for traj in self.data:
            mean_coords[traj[0]][:,:dim] = np.mean(traj[1][startframe:], axis=0)
            self.feg[traj[0]][:,:dim] = - self.fep_k * (
                mean_coords[traj[0]][:,:dim] - self.path[traj[0]][:,:dim]
            )

        if move_nodes_to_mean:
            if self.verbose:
                print(' >>> FENEB: recentering path nodes on trajectory averages')
            tmp = self.path.copy()
            self.path = [tmp[0]]
            for i, node in enumerate(mean_coords[1:-1]):
                if node.sum() != 0.0:
                    if active_relaxed is not None:
                        self.path.append(tmp[i+1])
                        self.path[-1][active_relaxed] = node[active_relaxed]
                    else:
                        self.path.append(node)
                else:
                    self.path.append(tmp[i+1])
            self.path.append(tmp[-1])
        elif self.verbose:
            print(' >>> FENEB: path nodes not recentered')

        return self.feg

    def load_data(self, datapath: str, traj_filename: str) -> List[List[np.ndarray]]:
        """Load data for subsequent FENEB step, uses the 'MDTraj' package as 'ASE' cannot read the DCD file format

        Assumes the following folder structur: 
            <datapath>/<ImageIDs>/<traj_filename>, where 'ImageIDs' are Integers in range(0, N_images)

        Args:
            datapath: path to simulation data of current optimization step
            traj_filename: filename of data file

        Returns:
            data: list of lists with xyz coordinates for trajectories corresponding to path nodes
        """
        if self.load_from_dcd:
            try:
                import mdtraj as md
            except ImportError as e:
                print(f"Failed to import MDTraj: {e}")

        self.data = []
        for i in range(self.nimages):
            path = os.path.join(datapath, f"{i}/{traj_filename}")
            if os.path.isfile(path):
                if self.load_from_dcd:
                    # mdtraj stores data in nm!
                    if self.convert_units:
                        self.data.append([i, md.load(path, top=self.top).xyz * 10.0 / BOHR_to_ANGSTROM]) # coords in a.u.
                    else:
                        self.data.append([i, md.load(path, top=self.top).xyz * 10.0]) # coords in angstrom
                else:
                    self.data.append([i, np.loadtxt(path)])
                if self.verbose:
                    print(f" >>> FENEB: Loaded data from {path}")
            else: 
                if self.verbose:
                    print(f" >>> FENEB: No data in {path}")
    
        # Sort the simulations based on their image ID
        self.data.sort(key=lambda x: x[0])
        return [xyz for xyz in self.data]

    def write_nodes_to_xyz(self):
        """ Saves one `.xyz` file for each path node
        """
        for i, coords in enumerate(self.path):     
            if self.convert_units:
                self.initial.set_positions(coords * BOHR_to_ANGSTROM)
            else:
                self.initial.set_positions(coords)
            io.write(f"{i}.xyz", self.initial)


def confine_md_to_node(
    coords: np.ndarray, 
    reference_coords: np.ndarray, 
    k: float,
    convert_units: bool=True,
) -> (float, np.ndarray):
    """Harmonic confinement of `coords` to `reference_coords`

    Args:
        coords: Coordinates
        reference_coords: Reference coordinates with shape(coords)
        k: force constant of confinement
        convert_units: if True, convert k from kJ/molA^2 to Ha/a_0^2
        
    Returns:
        conf_energy: confinement energy
        conf_forces: confinement forces
    """
    if convert_units:
        k = k * BOHR_to_ANGSTROM * BOHR_to_ANGSTROM / atomic_to_kJmol
    
    diff = coords - reference_coords
    conf_energy = (k / 2.) * np.sum(np.square(diff))
    conf_forces = k * diff
        
    return conf_energy, conf_forces
