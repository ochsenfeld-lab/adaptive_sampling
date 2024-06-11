import numpy as np
from typing import List, Tuple

from adaptive_sampling.colvars import CV
from adaptive_sampling import units

class Harmonic_Constaint():
    """ Harmonic constraint for collective variables
    """
    def __init__(
        self, 
        the_md: object,
        force_constants: List[float], 
        equil_position: List[float],
        colvars: dict,
        outputfile: str="constraints_traj.dat",
        output_stride: int=1, 
    ):
        self.the_md = the_md
        self.equil_positions = np.asarray(equil_position) if hasattr(equil_position, "__len__") else np.asarray([equil_position])
        self.force_constants = np.asarray(force_constants) if hasattr(force_constants, "__len__") else np.asarray([force_constants])
        self.force_constants /= units.atomic_to_kJmol # convert kJ/mol to atomic units
        self.outputfile = outputfile
        self.output_stride = output_stride

        # init collective variable
        self.the_cv = CV(self.the_md, requires_grad=True)
        self.colvars = colvars 
        if len(self.colvars) != len(self.force_constants):
            raise(" >>> Harmonic_Constraint: Number of colvars does not match number of constraints")


    def step_bias(self, **kwargs):
        """ Applies harmonic constraint to `colvars`

        Returns:
            bias_force: bias force array with shape(Natoms,)
        """
        cvs, grad_cvs, cv_types = self.get_cvs(**kwargs)
        conf_energy = []
        conf_forces = np.zeros_like(grad_cvs[0])
        for k, cv_0, cv, grad_cv, cv_type in zip(self.force_constants, self.equil_positions, cvs, grad_cvs, cv_types):
            k, cv_0 = self.unit_conversion(k, cv_0, cv_type)
            d = cv - cv_0
            conf_energy.append(0.5 * k * d * d)
            conf_forces += k * d * grad_cv

        # output
        md_step = self.the_md.get_sampling_data().step
        if md_step%self.output_stride == 0:
            self.print_conf(md_step, cvs, conf_energy, cv_types)

        return conf_forces


    def get_cvs(self, **kwargs) -> Tuple[list, list, list]:
        """ get state of all collective variables

        Returns:
            cvs: state of the CVs
            grad_cvs: gradients of the CVs
            cv_types: type of the CVs (needed for correct unit conversion)
        """
        cvs, grad_cvs, cv_types = [], [], []

        for cv, cv_def in self.colvars.items():
            cv, grad_cv = self.the_cv.get_cv(cv, cv_def, **kwargs)
            cvs.append(cv)
            grad_cvs.append(grad_cv)
            cv_types.append(self.the_cv.type)
        
        return cvs, grad_cvs, cv_types
    

    def unit_conversion(self, k: float, x0: float, type: str) -> Tuple[float, float]:
        """Units conversion for angles and distances
        
        Args:
            k: force constant
            x0: equilibrium position
            type: type of CV
        
        Returns:
            k: force constant in atomic units
            x0: equilibrium position in atomic units
        """
        if type == "distance":
            x0_bohr = x0 / units.BOHR_to_ANGSTROM
            k_bohr = k * units.BOHR_to_ANGSTROM * units.BOHR_to_ANGSTROM
            return k_bohr, x0_bohr
        
        elif type == "angle":
            x0_rad = x0 / units.DEGREES_per_RADIAN
            k_rad = k * units.DEGREES_per_RADIAN * units.DEGREES_per_RADIAN
            return k_rad, x0_rad
        
        return k, x0


    def print_conf(self, md_step, cvs, epots, cv_types):
        """ print confinments of current step to `self.outputfile` """
        if md_step==0:
            with open(self.outputfile, "w") as f:
                f.write(f"# Step\t")
                for i, type in enumerate(cv_types):
                    f.write(f"{f'{type}_{i}':12s} {f'E_conf_{i}':12s} ")
                f.write("\n")

        with open(self.outputfile, "a") as f:
            f.write(f"  {md_step}\t")
            for i, (cv, epot) in enumerate(zip(cvs, epots)):
                f.write(f"{cv:12.6f} {epot:12.6f}")
            f.write("\n")
