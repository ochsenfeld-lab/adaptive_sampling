import numpy as np
from .reactor import Reactor
from ..sampling_tools.amd import aMD
from ..units import *

class Ultrareactor(aMD):

    def __init__(
        self,
        k_conf: float,
        relative_soft_box_size: float,
        *args,
        mode: str = "GaHRD_lower", #aHRD, GaHRD, SaHRD
        qm_confinement: str = "spherical",
        **kwargs
    ):
        # Definition of reaction coordinate (Collective Variable) ! This is just a dummy CV because of the code's architecture requires one.
        CV_type         = 'distance'
        atom_indices    = [0,1]
        min_xi          = 0.5        # A
        max_xi          = 10.0       # A
        bin_width       = 0.5        # A
        cv = [[CV_type, atom_indices, min_xi, max_xi, bin_width]]

        aMD.__init__(self, cv_def=cv, confine=False, verbose=False, *args, **kwargs)

        self.mode = mode.lower()
        if self.mode == "ahrd":
            self.amd_method = "amd"
        elif self.mode == "gahrd_lower":
            self.amd_method = "gamd_lower"
        elif self.mode == "gahrd_upper":
            self.amd_method = "gamd_upper"
        elif self.mode == "sahrd":
            self.amd_method = "samd"

        self.k_conf = k_conf*np.power(BOHR_to_ANGSTROM,2.0)/(atomic_to_kJmol*kJ_to_kcal)

        self.relative_soft_box_size = relative_soft_box_size
        self.qm_confinement = qm_confinement.lower()

    def _soft_wall_bias(
            self
    ) -> np.ndarray:
        md_state = self.the_md.get_sampling_data()
        import numpy as np

        # Inputs (examples)
        coords = md_state.coords.reshape(md_state.natoms, 3)          # shape (N_atoms, 3)
        
        unit_cell_dimensions =  np.asarray(self.the_md.calculator.mm_theory.topology.getUnitCellDimensions()._value) / BOHR_to_NANOMETER # Bohr
        center = np.copy(unit_cell_dimensions) / 2
        r_t = self.relative_soft_box_size * center 
        masses = md_state.mass

        # Displacement vector
        dx = coords - center                    # shape (N_atoms, 3)

        # Distance from center
        d = np.abs(dx)                # shape (N_atoms, 3)

        # Difference to threshold
        delta = np.maximum(0, d - r_t)  # shape (N_atoms, 3)

        # Multiply by force constant, masses, and direction
        grad = masses[:, np.newaxis] * self.k_conf * delta * np.sign(dx)  # shape (N_atoms, 3)
        grad[~self.the_md.calculator.xatom_mask] = np.array((0, 0, 0))
        return grad.reshape(-1) # shape (N_atoms * 3,)

    def _spherical_bias(
        self, 
    ) -> np.ndarray:
        
        md_state = self.the_md.get_sampling_data()
        bias_force = np.zeros_like(md_state.forces)
        bias_pot = md_state.epot

        unit_cell_dimensions =  np.asarray(self.the_md.calculator.mm_theory.topology.getUnitCellDimensions()._value) / BOHR_to_NANOMETER # Bohr
        center = unit_cell_dimensions / 2
        self.radius = self.relative_soft_box_size * np.min(center)

        for i in self.the_md.calculator.qmatoms:
            xx = md_state.coords[3*i+0] - center[0]
            yy = md_state.coords[3*i+1] - center[1]
            zz = md_state.coords[3*i+2] - center[2]
            r  = np.sqrt(xx*xx+yy*yy+zz*zz)
            mass = md_state.mass[i]
            dbase = 0.e0
            rconf = 0.e0

            if r == 0.e0:
                dbase = 0.e0
            else:
                rconf = np.max([0.e0,r-self.radius])
                bias_pot += 0.5e0 * self.k_conf * np.power(rconf,2.e0) * mass
                dbase = self.k_conf * rconf/r * mass

            bias_force[i*3+0] += xx * dbase
            bias_force[i*3+1] += yy * dbase
            bias_force[i*3+2] += zz * dbase
            
        return bias_force

    def step_bias(self, *args, **kwargs):
        amd_bias = aMD.step_bias(self, write_output=False, write_traj=False,*args, **kwargs)
        if self.qm_confinement == "spherical":
            amd_bias += self._spherical_bias()
        elif self.qm_confinement == "box":
            amd_bias += self._soft_wall_bias()
        else:
            raise ValueError(f"Confinement '{self.qm_confinement}' unknown. Choose between ['spherical', 'box']")
        return  amd_bias
    


