import os
import numpy as np
from typing import List, Union
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
    """
    def __init__(
        self,
        fep_k: float,
        neb_k: float,
        endpoints: List[str],
        old_path: np.ndarray=None,
        nimages: int=10,
        conf_spring: float=0.1,
        maxiter_spring: int=1000,
        active: List[int]=None,
        idpp: bool=False,
        load_from_dcd: bool=True,
        top: str=None,
        write_xyz: bool=False,
        verbose: bool=False,
    ):
        self.neb_k = neb_k
        self.fep_k = fep_k 
        self.verbose = verbose
        self.active = active
        self.conf_spring = conf_spring
        self.maxiter_spring = maxiter_spring
        self.nimages = nimages

        self.load_from_dcd = load_from_dcd
        if self.load_from_dcd:
            try:
                import mdtraj as md
            except ImportError as e:
                print(f"Failed to import MDTraj: {e}")
            
            self.top = top
            if self.top == None:
                print('WARNING: you need to give a topology (e.g. pdb or prmtop) for dcd import!')
            if self.verbose:
                print(f'INFO: Topology for dcd import: {self.top}')
        
        self.optstep = 0

        self.initial = io.read(endpoints[0]) # always get on ase.Atoms object for io operations
        if self.active == None:
            self.active = [i for i in range(len(self.initial.get_atomic_numbers()))]
        self.natoms = len(self.initial.get_atomic_numbers())

        if old_path == None:
            self.final   = io.read(endpoints[1])
            self.path = self.make_path(idpp, write_xyz=write_xyz)
        else:
            self.path = np.load(old_path)
            self.path = [node for node in self.path]

    def step(self, feg: np.ndarray=None, uncoupled_band_opt: bool=True) -> dict:
        """NEB optimisation step (Steepest Descent)

        Args:
            feg: Cartesian free energy gradient in shape (N_nodes, M_atoms, 3)
            uncoupled_band_opt: full optimization of spring force
        """
        self.optstep += 1
        confs_fes, scaling_factors = [], []
        for i, _ in enumerate(self.path[1:-1]):
            
            # tanget vector to path
            tau = self.tangent_vector([
                self.path[i][self.active],
                self.path[i+1][self.active],
                self.path[i+2][self.active],
            ])      

            # SD step along free energy gradient
            feg_i = np.asarray([feg[i+1]]).flatten()             
            force = feg_i - np.dot(feg_i, tau) * tau
            
            if not uncoupled_band_opt:
                force -= self.neb_k * np.dot(
                    (self.path[i+2][self.active].flatten()-self.path[i+1][self.active].flatten())-
                    (self.path[i+1][self.active].flatten()-self.path[i+0][self.active].flatten()), tau) * tau
                
            force = force.reshape((self.natoms,3))

            force_norm_max, scaling_factor = self._get_scaling_factor(force)
            if force_norm_max > 0.0:
                self.path[i+1][self.active] -= scaling_factor * force[self.active] / force_norm_max 
            else:
                if self.verbose:
                    print(' >>> WARNING: Maximum norm of FENEB force is zero')

            confs_fes.append(force_norm_max)
            scaling_factors.append(scaling_factor)

        if uncoupled_band_opt:
            conf_spring, iter_spring = self.opt_node_spacing()
        else: 
            conf_spring, iter_spring = 0.0, 0.0

        if self.verbose:
            if uncoupled_band_opt:
                print(
                    f'Iter {self.optstep:3d}:\tMax(Forces) = {np.max(confs_fes):3.6f} (idx: {np.argmax(confs_fes)+1:3d}),\tMax(F_spring) = {np.max(conf_spring):3.6f} (Iters: {iter_spring:3d})'
                )
            else:
                print(
                    f'Iter {self.optstep:3d}:\tMax(Forces) = {np.max(confs_fes):3.6f} (idx: {np.argmax(confs_fes)+1:3d})'
                )

        results = {
            'Iter': self.optstep, 
            'Max_force': np.max(confs_fes), 
            'Max_force_idx': np.argmax(confs_fes)+1,
            'Max_force_string': conf_spring,
        }
        return results

    def opt_node_spacing(self, verbose: bool=False):
        """ full optimization of spring force """

        conf_spring = 9999
        iter_spring = 0

        while True:
            
            tmp_path = self.path.copy()
            confs_spring = []
            for i, _ in enumerate(self.path[1:-1]):
                
                tau = self.tangent_vector([
                    self.path[i][self.active],
                    self.path[i+1][self.active],
                    self.path[i+2][self.active],
                ])      
                
                fspring = self.neb_k * np.dot(
                    (self.path[i+2][self.active].flatten()-self.path[i+1][self.active].flatten())-
                    (self.path[i+1][self.active].flatten()-self.path[i+0][self.active].flatten()), tau) * tau
                fspring = fspring.reshape((self.natoms,3))

                conf, spring_scaling = self._get_scaling_factor(fspring)
                confs_spring.append(conf)

                if conf > 0.0:
                    tmp_path[i+1][self.active] += spring_scaling * fspring[self.active] / conf    
                else:
                    if self.verbose:
                        print(' >>> WARNING: Maximum norm of FENEB force is zero')

            iter_spring += 1
            conf_spring = np.max(confs_spring)
            self.path = tmp_path.copy()

            if conf_spring < self.conf_spring:
                if verbose:
                    print(f" >>> INFO: Spring optimization converged after {iter_spring} iterations.")
                break
            
            if self.maxiter_spring <= iter_spring:
                if verbose:
                    print(f" >>> INFO: Final spring force after {iter_spring} steps: {conf_spring}.")
                break

        return conf_spring, iter_spring

    @staticmethod
    def tangent_vector(image):
        """ tanget vector to path """
        x_prev = image[1] - image[0]
        x_prev /= np.linalg.norm(x_prev)

        x_next = image[2] - image[1]
        x_next /= np.linalg.norm(x_next)

        tau = x_prev + x_next
        tau /= np.linalg.norm(tau)    
        return tau.flatten()       

    @staticmethod
    def _get_scaling_factor(forces):
        norm_max = np.max(np.linalg.norm(forces.reshape((int(len(forces.flatten())/3),1,3)), axis=2))
        if norm_max > 5.0:
            scaling_factor = 0.005 / BOHR_to_ANGSTROM
        elif norm_max > 2.5:
            scaling_factor = 0.002 / BOHR_to_ANGSTROM
        else:
            scaling_factor = 0.0001 / BOHR_to_ANGSTROM
        return norm_max, scaling_factor

    def make_path(self, idpp: bool, write_xyz: bool=True) -> List[np.ndarray]:
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
            from ase.constraints import FixAtoms
            apply_constraint = True
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
        
        if write_xyz:
            self.write_nodes_to_xyz()

        return [image.get_positions() / BOHR_to_ANGSTROM for image in images]

    def calc_feg(self, startframe: int=0, move_nodes_to_mean: bool=True, dim: int=3) -> np.ndarray:
        """Get free energy gradient for Umbrella Integration

        Args:
            move_nodes_to_mean: set nodes to mean coords of Umbrella
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
            self.path = []
            for i, node in enumerate(mean_coords):
                if node.sum() != 0.0:
                    self.path.append(node)
                else:
                    self.path.append(tmp[i])
        elif self.verbose:
            print(' >>> FENEB: path nodes not recentered')

        return self.feg

    def load_data(self, datapath: str, traj_filename: str) -> List[List[Union[int, np.ndarray]]]:
        """Load data for subsequent FENEB step

        Assumes the following folder structur: 
            <datapath/<ImageIDs>/<traj_filename>, where `ImageIDs` can be Integers in range(0, N_nodes)

        Args:
            datapath: path to data of current optimization step
            traj_filename: filename of data file

        Returns:
            data: list with [ImageID, xyz data] for all subfolders of `datapath` named `ImageID`
        """

        try:
            import mdtraj as md
        except ImportError as e:
            print(f"Cannot import MDTraj: {e}")

        self.data = []
        for i in range(10):
            path = os.path.join(datapath, f"{i}/{traj_filename}")
            if os.path.isfile(path):
                if self.load_from_dcd:
                    self.data.append([i, md.load(path, top=self.top).xyz])
                else:
                    self.data.append([i, np.loadtxt(path)])
                if self.verbose:
                    print(f" >>> FENEB: Loaded data from {os.path.join(path, traj_filename)}.")
            else: 
                if self.verbose:
                    print(f" >>> FENEB: No data in {path}.")
    
        # Sort the simulations based on their image ID
        self.data.sort(key=lambda x: x[0])

        return [xyz for xyz in self.data]

    def write_nodes_to_xyz(self):
        """ Saves one `.xyz` file for each path node
        """
        for i, coords in enumerate(self.path):     
            self.initial.set_positions(coords * BOHR_to_ANGSTROM)
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
