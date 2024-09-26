import numpy as np
from .reactor import Reactor
from ..sampling_tools.amd import aMD
from ..units import *

class Hyperreactor(Reactor, aMD):

    def __init__(
        self,
        r_min: float,
        r_max: float,
        t_total: float,
        k_conf: float,
        *args,
        mode: str = "GaHRD_lower", #aHRD, GaHRD, SaHRD
        **kwargs
):
        # Definition of reaction coordinate (Collective Variable)
        CV_type         = 'distance'
        atom_indices    = [0,1]
        min_xi          = 0.5        # A
        max_xi          = 10.0        # A
        bin_width       = 0.5       # A
        cv = [[CV_type, atom_indices, min_xi, max_xi, bin_width]]

        Reactor.__init__(self, *args, **kwargs)
        aMD.__init__(self, confine=False, cv_def=cv, *args, **kwargs)

        self.mode = mode.lower()
        if self.mode == "ahrd":
            self.amd_method = "amd"
        elif self.mode == "gahrd_lower":
            self.amd_method = "gamd_lower"
        elif self.mode == "gahrd_upper":
            self.amd_method = "gamd_upper"
        elif self.mode == "sahrd":
            self.amd_method = "samd"
        
        self.r_max = r_max/BOHR_to_ANGSTROM
        self.r_min = r_min/BOHR_to_ANGSTROM
        self.t_total = t_total
        self.k_conf = k_conf*np.power(BOHR_to_ANGSTROM,2.0)/(atomic_to_kJmol*kJ_to_kcal)

        self.radius = self.r_max

    def _spherical_bias(
        self, 
    ) -> np.ndarray:
        
        md_state = self.the_md.get_sampling_data()
        bias_force = np.zeros_like(md_state.forces)
        bias_pot = md_state.epot

        t = md_state.step * md_state.dt
        self.radius = min(self.r_max + (self.r_max - self.r_min) * np.sin(np.pi/2*np.cos(t/self.t_total*2*np.pi)), self.r_max)

        for i in range(self.the_md.natoms):
            xx = self.the_md.coords[3*i+0]
            yy = self.the_md.coords[3*i+1]
            zz = self.the_md.coords[3*i+2]
            r  = np.sqrt(xx*xx+yy*yy+zz*zz)
            mass = self.the_md.mass[i]

        if r == 0.e0:
            dbase = 0.e0
        else:
            maxr = np.max([0,r-self.radius])
            bias_pot += 0.5e0 * self.k_conf * np.power(maxr,2.e0) * mass

            dbase = self.k_conf * maxr/r * mass
        bias_force[i*3+0] += xx * dbase
        bias_force[i*3+1] += yy * dbase
        bias_force[i*3+2] += zz * dbase

        return bias_force

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):
        amd_bias = aMD.step_bias(self)

        return  amd_bias + self._spherical_bias()
    


