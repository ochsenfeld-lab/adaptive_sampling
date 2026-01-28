import numpy as np
from .. import units

class Sphere():

    def __init__(
        self, 
        the_md: object=None, 
        k_conf: float = None, 
        r_min: float = 0.0, 
        r_max: float = 0.0, 
        t_period: float = None, 
        t_shift: float = 0.0,
        confinement_method: str = 'smooth-step'
    ):

        if the_md is None:
            raise ValueError(" >>> ERROR: Molecular dynamics object has to be provided for spherical confinment.")
        self.the_md = the_md

        # confinement force conststant in atomic units
        if k_conf is None or k_conf <= 0:
            raise ValueError(" >>> ERROR: Force constant for spherical confinment has to be provided and > 0.")
        self.k_conf_max = k_conf * np.power(units.BOHR_to_ANGSTROM,2.0) / (units.atomic_to_kJmol * units.kJ_to_kcal)
        self.k_conf_min = self.k_conf_max / 2.e0 
        
        # period for contraction of spherical confinment in fs
        self.t_period = t_period
        self.t_expand = 3.e0 / 4.e0 * self.t_period
        self.t_contract = 1.e0 / 4.e0 * self.t_period
        self.t_shift = t_shift

        # minimum and maximum radius for spherical confinment in atomic units
        if self.r_min is 0.0 or self.r_max is 0.0:
            raise ValueError(" >>> ERROR: Both minimum and maximum radius for spherical confinment have to be provided and > 0.")
        self.r_min = r_min * units.BOHR_to_ANGSTROM
        self.r_max = r_max * units.BOHR_to_ANGSTROM

        # confinement method
        if confinement_method.lower() not in ['smooth-step', 'smooth', 'step', 'constant']:
            raise ValueError(" >>> ERROR: Invalid confinment method. Choose from 'smooth-step', 'smooth', 'step' or 'constant'.")
        self.confinement_method = confinement_method.lower()        
        if self.confinement_method == 'constant' and self.t_period is not None:
            print(" >>> WARNING: 'constant' confinement method does not depend on time period. Setting time period to None.")
            self.t_period = None

        self.bias_pot = 0.0

    def step_bias(self) -> np.array:
        """Calculates the spherical confinement potential and force.

        Args:
            r (float): Distance from the center in atomic units.
            t (float): Current time in fs.
        Returns:
            f_conf: Confinement force in atomic units.
        """
        md_state = self.the_md.get_sampling_data()
        bias_force = np.zeros_like(md_state.coords)
        
        t = md_state.step * md_state.dt - self.t_shift  # current time in fs, starting at time shift

        # start with confinement after time shift
        if t > 0.e0:
            # get atom wise confinement forces
            for i in self.atoms:
                xx = md_state.coords[3*i+0]
                yy = md_state.coords[3*i+1]
                zz = md_state.coords[3*i+2]
                r  = np.sqrt(xx*xx+yy*yy+zz*zz)
                mass = self.the_md.mass[i]

                if self.confinement_method == "constant":
                    maxr = np.max([0,r-self.r_max])
                    self.bias_pot += 0.5e0 * self.k_conf_max * np.power(maxr,2.e0) * mass
                    dbase = self.k_conf_max * maxr / r * mass

                if self.confinement_method == "step":
                    f = np.heaviside(np.floor(t / (self.t_contract + self.t_expand)) 
                        - t / (self.t_contract + self.t_expand) 
                        + self.t_expand / (self.t_contract + self.t_expand))
                    U_max = mass * self.k_conf_max / 2.e0 * np.power((r - self.r_max),2) * np.heaviside(r - self.r_max)
                    U_min = mass * self.k_conf_min / 2.e0 * np.power((r - self.r_min),2) * np.heaviside(r - self.r_min)
                    self.bias_pot += f * U_max + (1 - f) * U_min

                    dbase = (f * self.k_conf_max * mass * (r - self.r_max) / r
                        + (1 - f) * self.k_conf_min * mass * (r - self.r_min) / r)

                elif self.confinement_method == "smooth-step":
                    radius = np.min(self.r_max + (self.r_max - self.r_min) * np.sin(np.pi/2*np.cos(t/(self.t_expand + self.t_contract)*2*np.pi)) , self.r_max)   
                    if r == 0.e0:
                        dbase = 0.e0
                    else:
                        maxr = np.max([0,r-radius/units.BOHR_to_ANGSTROM])
                        self.bias_pot += 0.5e0 * self.k_conf_max * np.power(maxr,2.e0) * mass

                        dbase = self.k_conf_max * maxr/r * mass

                elif self.confinement_method == "smooth":
                    radius = self.r_min + (self.r_max - self.r_min) * (1e0 + np.cos(t/(self.t_expand + self.t_contract)*2*np.pi))
                    if r == 0.e0:
                        dbase = 0.e0
                    else:
                        maxr = np.max([0,r-radius/units.BOHR_to_ANGSTROM])
                        self.bias_pot += 0.5e0 * self.k_conf_max * np.power(maxr,2.e0) * mass

                        dbase = self.k_conf_max * maxr / r * mass

                bias_force[3*i+0] += dbase * xx
                bias_force[3*i+1] += dbase * yy
                bias_force[3*i+2] += dbase * zz
        
        return bias_force
        