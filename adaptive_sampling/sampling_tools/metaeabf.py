import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, diff
from .eabf import eABF
from .metadynamics import WTM
from ..units import *


class WTMeABF(eABF, WTM, EnhancedSampling):
    """Well-Tempered Metadynamics extended-system Adaptive Biasing Force method
       
       see: Fu et. al., J. Phys. Chem. Lett. (2018); https://doi.org/10.1021/acs.jpclett.8b01994

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased using a combination of ABF and Metadynamics.

    Args:
        ext_sigma: thermal width of coupling between collective and extended variable
        ext_mass: mass of extended variable in atomic units
        nfull: Number of force samples per bin where full bias is applied, 
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CV (can be Bohr, Degree, or None)
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        friction: friction coefficient for Lagevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance, 
                         if False, forces are calculated from sum of Gaussians in every step 
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
        
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abf_forces = np.zeros_like(self.bias)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)
        self._propagate()

        mtd_forces = self.get_wtm_force(self.ext_coords)
        bias_force = self._extended_dynamics(xi, delta_xi) #, self.hill_std)

        if (self.ext_coords <= self.maxx).all() and (
            self.ext_coords >= self.minx
        ).all():

            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink[1], bink[0]] += 1

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.ext_hist[bink[1], bink[0]] > self.nfull
                    else self.ext_hist[bink[1], bink[0]] / self.nfull
                )

                # apply bias force on extended variable
                (
                    self.abf_forces[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    self.var_force[i][bink[1], bink[0]],
                ) = welford_var(
                    self.ext_hist[bink[1], bink[0]],
                    self.abf_forces[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    self.ext_k[i] 
                    * diff(self.ext_coords[i], xi[i], self.cv_type[i]),
                )
                self.ext_forces -= (
                    ramp * self.abf_forces[i][bink[1], bink[0]] - mtd_forces[i]
                )

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # xi-conditioned accumulators for CZAR
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]], self.cv_type[i])
                self.correction_czar[i][bink[1], bink[0]] += self.ext_k[i] * dx

        self._up_momenta()

        self.traj = np.append(self.traj, [xi], axis=0)
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        if md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"metaforce {i}"] = self.bias[i]
                    output[f"abf force {i}"] = self.abf_forces[i]
                    # TODO: give variance of CZAR not bias
                    output[f"var force {i}"] = self.var_force[i]
                    output[f"czar force {i}"] = self.czar_force[i]
                output[f"metapot"] = self.metapot

                self.write_output(output, filename="wtmeabf.out")
                self.write_restart()

        return bias_force

    def shared_bias(self):
        """TODO"""
        pass

    def write_restart(self, filename: str = "restart_wtmeabf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            force=self.bias,
            var=self.var_force,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
            abf_force=self.abf_forces,
            center=self.center,
            metapot=self.metapot,
        )

    def restart(self, filename: str = "restart_wtmeabf"):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.bias = data["force"]
        self.var_force = data["var"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces = data["abf_force"]
        self.center = data["center"].tolist()
        self.metapot = data["self.metapot"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")


    def write_traj(self):
        """save trajectory for post-processing"""

        data = {}
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.ext_traj[:, i] *= DEGREES_per_RADIAN
            elif self.cv_type[i] == "distance":
                self.ext_traj[:, i] *= BOHR_to_ANGSTROM
            data[f"lambda{i}"] = self.ext_traj[:, i]
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
