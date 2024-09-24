import numpy as np
from .reactor import Reactor
from ..units import *

class Nanoreactor(Reactor):

    def __init__(
        self,
        boost_temperature: float,
        r_max: float,
        r_min: float,
        k_max: float,
        k_min: float,
        t_expand: float,
        t_contract: float,
        *args,
        confinement_method : str,
        **kwargs
):
        super().__init__(*args, **kwargs)
        self.confinement_method = confinement_method.lower()
        self.the_md.target_temp = boost_temperature

        self.k_conf_max = k_max*np.power(BOHR_to_ANGSTROM,2.0)/(atomic_to_kJmol*kJ_to_kcal)
        self.k_conf_min = k_min*np.power(BOHR_to_ANGSTROM,2.0)/(atomic_to_kJmol*kJ_to_kcal)
        self.r_max = r_max/BOHR_to_ANGSTROM
        self.r_min = r_min/BOHR_to_ANGSTROM
        self.t_expand = t_expand
        self.t_contract = t_contract

    def _spherical_bias(
        self, 
        **kwargs
    ) -> np.ndarray:

        md_state = self.the_md.get_sampling_data()
        bias_force = np.zeros_like(md_state.forces)
        bias_pot = 0.e0

        # determine radius at present MD time step
        t = md_state.step * md_state.dt # in femtoseconds

        for i in self.atoms:
            xx = self.the_md.coords[3*i+0]
            yy = self.the_md.coords[3*i+1]
            zz = self.the_md.coords[3*i+2]
            r  = np.sqrt(xx*xx+yy*yy+zz*zz)
            mass = self.the_md.mass[i]

            if self.confinement_method == "step":
                f = np.heaviside(np.floor(t / (self.t_contract + self.t_expand)) 
                                - t / (self.t_contract + self.t_expand) 
                                + self.t_expand / (self.t_contract + self.t_expand))
                U_max = mass * self.k_conf_max / 2.e0 * np.power((r - self.r_max),2) * np.heaviside(r - self.r_max)
                U_min = mass * self.k_conf_min / 2.e0 * np.power((r - self.r_min),2) * np.heaviside(r - self.r_min)
                bias_pot += f * U_max + (1 - f) * U_min

                dbase = (f * self.k_conf_max * mass * (r - self.r_max) / r
                        + (1 - f) * self.k_conf_min * mass * (r - self.r_min) / r)

            elif self.confinement_method == "smooth-step":
                radius = np.min(self.r_max + (self.r_max - self.r_min) * np.sin(np.pi/2*np.cos(t/(self.t_expand + self.t_contract)*2*np.pi)) , self.r_max)   
                if r == 0.e0:
                    dbase = 0.e0
                else:
                    maxr = np.max([0,r-radius/BOHR_to_ANGSTROM])
                    bias_pot += 0.5e0 * self.k_conf * np.power(maxr,2.e0) * mass

                    dbase = self.k_conf * maxr/r * mass

            elif self.confinement_method == "smooth":
                radius = self.r_min + (self.r_max - self.r_min) * (1e0 + np.cos(t/(self.t_expand + self.t_contract)*2*np.pi))
                if r == 0.e0:
                    dbase = 0.e0
                else:
                    maxr = np.max([0,r-radius/BOHR_to_ANGSTROM])
                    bias_pot += 0.5e0 * self.k_conf * np.power(maxr,2.e0) * mass

                    dbase = self.k_conf * maxr/r * mass

            bias_force[i*3+0] += xx * dbase
            bias_force[i*3+1] += yy * dbase
            bias_force[i*3+2] += zz * dbase

        return bias_force


    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):
        spherical_bias = self._spherical_bias()

        return spherical_bias
    

    


