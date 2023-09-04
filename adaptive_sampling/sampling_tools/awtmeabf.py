import numpy as np
from .enhanced_sampling import EnhancedSampling
from .metaeabf import WTMeABF
from .amd import aMD
from .utils import welford_var, diff, cond_avg
from ..processing_tools.thermodynamic_integration import integrate
from ..units import *


class aWTMeABF(WTMeABF, aMD, EnhancedSampling):
    """Accelerated Well-Tempered Metadynamics extended-system Adaptive Biasing Force Method

    see: Chen et. al., J. Chem. Theory Comput. (2021); https://doi.org/10.1021/acs.jctc.1c00103

    The collective variable (CV) is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased using a combination of ABF and metadynamics.
    The dynamics of the pysical system is biased with CV-independend an GaMD boost potential

    Args:
        ext_sigma: thermal width of coupling between collective and extended variable
        ext_mass: mass of extended variable in atomic units

        nfull: Number of force samples per bin where full bias is applied,
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        friction: friction coefficient for Lagevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CV (can be Bohr, Degree, or None)
        amd_parameter: upper limit of standard deviation of boost potential (sigma_0 for GaMD)
        init_step: initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps: equilibration steps, min, max and var of potential energy is still updated
                          force constant of coupling is calculated from previous steps
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        do_wtm: if False, no metadynamics potential is applied (Gaussian-accelerated eABF)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance,
                         if False, forces are calculated from sum of Gaussians in every step
        amd_method: "amd": use accelerated MD
                    "gamd_lower": use lower bound for GaMD boost
                    "gamd_upper: use upper bound for GaMD boost
                    "samd": use sigmoid accelerated MD
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing output
    """

    def __init__(self, *args, qm_boost: bool = False, do_wtm: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_wtm = do_wtm
        self.qm_boost = qm_boost

    def step_bias(
        self, 
        write_output: bool = True, 
        write_traj: bool = True, 
        stabilize: bool = False,
        stabilizer_threshold: float = None,
        output_file: str = 'awtmeabf.out',
        traj_file: str = 'CV_traj.dat', 
        restart_file: str = 'restart_awtmeabf',
        **kwargs
    ) -> np.ndarray:
        """Apply accelerated WTM-eABF to MD simulations

        Args:
            write_output: if on-the-fly free energy estimate and restart files should be written
            write_traj: if CV and extended system trajectory file should be written
            stabilize: if stabilisation algorithm should be applied for discontinous CVs
            stabilizer_threshold: treshold for stabilisation of extended system
            output_file: name of the output file
            traj_file: name of the trajectory file
            restart_file: name of the restart file

        Returns:
            bias_force: Accelerated WTM-eABF biasing force of current step that has to be added to molecular forces
        """

        md_state = self.the_md.get_sampling_data()
        epot = md_state.epot
        if self.qm_boost:
            epot = md_state.epot - md_state.mm_epot
            self.amd_forces = np.copy(md_state.qm_force)
        else:
            epot = md_state.epot
            self.amd_forces = np.copy(md_state.forces)

        (xi, delta_xi) = self.get_cv(**kwargs)
        
        if stabilize and len(self.traj)>0:
            self.stabilizer(xi, threshold=stabilizer_threshold)

        self._propagate()
        bias_force = self._extended_dynamics(xi, delta_xi)

        if md_state.step < self.init_steps:
            self._update_pot_distribution(epot)

        else:
            if md_state.step == self.init_steps:
                self._calc_E_k0()

            # apply amd boost potential
            bias_force += self._apply_boost(epot)

            if md_state.step < self.equil_steps:
                self._update_pot_distribution(epot)
                self._calc_E_k0()

            else:
                # (WTM-)eABF bias on extended-variable only in production
                if self.do_wtm:
                    mtd_forces = self.get_wtm_force(self.ext_coords)

                if self._check_boundaries(self.ext_coords):

                    bink = self.get_index(self.ext_coords)
                    self.ext_hist[bink[1], bink[0]] += 1

                    for i in range(self.ncoords):

                        # linear ramp function
                        ramp = (
                            1.0
                            if self.ext_hist[bink[1], bink[0]] > self.nfull
                            else self.ext_hist[bink[1], bink[0]] / self.nfull
                        )

                        # apply (WTM-)eABF bias force on extended variable
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
                        self.ext_forces -= ramp * self.abf_forces[i][bink[1], bink[0]]

                        if self.do_wtm:
                            self.ext_forces += mtd_forces[i]

        # free energy reweighting
        if self._check_boundaries(xi):

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            # CZAR
            for i in range(self.ncoords):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]], self.cv_type[i])
                self.correction_czar[i][bink[1], bink[0]] += self.ext_k[i] * dx

            # aMD
            if md_state.step >= self.equil_steps:

                (
                    self.amd_c1[bink[1], bink[0]],
                    self.amd_m2[bink[1], bink[0]],
                    self.amd_c2[bink[1], bink[0]],
                ) = welford_var(
                    self.histogram[bink[1], bink[0]],
                    self.amd_c1[bink[1], bink[0]],
                    self.amd_m2[bink[1], bink[0]],
                    self.amd_pot,
                )

        self._up_momenta()

        self.traj = np.append(self.traj, [xi], axis=0)
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.temp.append(md_state.temp)
        self.amd_pot_traj.append(self.amd_pot)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        # write traj and output
        if md_state.step % self.out_freq == 0:

            if write_traj:
                self.write_traj(filename=traj_file)

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"abf force {i}"] = self.abf_forces[i]
                    output[f"czar force {i}"] = self.czar_force[i]
                output[f"metapot"] = self.metapot
                output[f"aMD corr"] = self.amd_corr

                self.write_output(output, filename=output_file)
                self.write_restart(filename=restart_file)

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):

        log_rho = np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(0 != self.histogram),
        )
        avg_force = cond_avg(self.correction_czar, self.histogram)

        self.amd_corr = -self.amd_c1 - self.amd_c2 / (
            2.0 * kB_in_atomic * self.equil_temp
        )

        if self.ncoords == 1:
            self.czar_force[0] = (
                -kB_in_atomic * self.equil_temp * np.gradient(log_rho[0], self.grid[0])
                + avg_force[0]
            )
            self.pmf[0, :], _ = integrate(
                self.czar_force[0][0],
                self.dx,
                equil_temp=self.equil_temp,
                method=method,
            )
            self.pmf += self.amd_corr
            self.pmf *= atomic_to_kJmol
            self.pmf -= self.pmf.min()

        else:
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
            self.czar_force[0] = (
                -kB_in_atomic * self.equil_temp * der_log_rho[1] + avg_force[0]
            )
            self.czar_force[1] = (
                -kB_in_atomic * self.equil_temp * der_log_rho[0] + avg_force[1]
            )
            if self.verbose:
                print(
                    " >>> Info: On-the-fly integration only available for 1D coordinates"
                )

    def shared_bias(self):
        """TODO"""
        pass

    def write_restart(self, filename: str = "restart_gaabf"):
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
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
            czar_corr=self.correction_czar,
            abf_force=self.abf_forces,
            center=self.center,
            metapot=self.metapot,
            amd_c1=self.amd_c1,
            amd_m2=self.amd_m2,
            corr=self.amd_corr,
            pot_count=self.pot_count,
            pot_var=self.pot_var,
            pot_std=self.pot_std,
            pot_m2=self.pot_m2,
            pot_avg=self.pot_avg,
            pot_min=self.pot_min,
            pot_max=self.pot_max,
            k0=self.k0,
        )   

    def restart(self, filename: str="restart_gaabf", restart_ext_sys: bool=False):
        """restart from restart file
        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram       = data["hist"]
        self.bias            = data["force"]
        self.var_force       = data["var"]
        self.m2_force        = data["m2"]
        self.ext_hist        = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces      = data["abf_force"]
        self.center          = data["center"].tolist()
        self.metapot         = data["metapot"]
        self.amd_c1          = data["amd_c1"]
        self.amd_m2          = data["amd_m2"]
        self.amd_corr        = data["corr"]
        self.pot_count       = data["pot_count"]
        self.pot_var         = data["pot_var"]
        self.pot_std         = data["pot_std"]
        self.pot_m2          = data["pot_m2"]
        self.pot_avg         = data["pot_avg"]
        self.pot_min         = data["pot_min"]
        self.pot_max         = data["pot_max"]
        self.k0              = data["k0"]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords  = data["ext_coords"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def write_traj(self, filename: str = 'CV_traj.dat'):
        """save trajectory for post-processing"""

        data = self._write_ext_traj()
        data[f"E_amd [H]"] = self.amd_pot_traj
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data, filename=filename)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.amd_pot_traj = [self.amd_pot_traj[-1]]
