import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var
from ..units import *


class GaMD(EnhancedSampling):
    """Gaussian-accelerated Molecular Dynamics
    
        see:
            aMD: Hamelberg et. al., J. Chem. Phys. 120, 11919 (2004); https://doi.org/10.1063/1.1755656
            GaMD: Miao et. al., J. Chem. Theory Comput. (2015); https://doi.org/10.1021/acs.jctc.5b00436

        Apply global boost potential to potential energy, that is independent of Collective Variables.

    Args:
        gamd_sigma0: upper limit of standard deviation of boost potential
        gamd_init_step: initial steps where no bias is applied to estimate min, max and var of potential energy
        gamd_equil_steps: equilibration steps, min, max and var of potential energy is still updated
                          force constant of coupling is calculated from previous steps
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        gamd_bound: "lower": use lower bound for GaMD boost
                    "upper: use upper bound for GaMD boost
                    "aMD": apply accelerated MD, gamd_sigma0 now defines the acceleration parameter alpha
        confine: is system should be confined to range of CV
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """

    def __init__(
        self,
        gamd_sigma0: float,
        gamd_init_steps: int,
        gamd_equil_steps: int,
        *args,
        gamd_bound: str = "lower",
        confine: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sigma0 = gamd_sigma0
        self.gamd_init_steps = gamd_init_steps
        self.gamd_equil_steps = gamd_equil_steps
        self.gamd_bound = gamd_bound.lower()
        self.confine = confine

        if self.verbose and gamd_bound.lower() == "amd":
            print(f" >>> Warning: Please use GaMD to obtain accurate free energy estimates!\n")

        self.pot_count = 0
        self.pot_var = 0.0
        self.pot_std = 0.0
        self.pot_m2 = 0.0
        self.pot_avg = 0.0
        self.pot_min = 0.0
        self.pot_max = 0.0
        self.k0 = 0.0
        self.E = 0.0

        self.gamd_pot = 0.0
        self.gamd_pot_traj = []

        self.gamd_c1 = np.zeros_like(self.histogram)
        self.gamd_c2 = np.zeros_like(self.histogram)
        self.gamd_m2 = np.zeros_like(self.histogram)
        self.gamd_corr = np.zeros_like(self.histogram)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        # get energy and forces
        bias_force = np.zeros_like(md_state.forces)
        self.gamd_forces = np.copy(md_state.forces)
        epot = md_state.epot

        if md_state.step < self.gamd_init_steps:
            self._update_pot_distribution(epot)

        else:
            if md_state.step == self.gamd_init_steps:
                self._calc_E_k0()

            # apply boost potential
            if self.gamd_bound.lower() == "amd":
                self.gamd_pot = np.square(self.E - epot) / (self.sigma0 + (self.E - epot))
                bias_force -= (
                    ((epot - self.E) * (epot - 2.0 * self.sigma0 - self.E)) / np.square(epot - self.sigma0 - self.E)
                ) * self.gamd_forces
            else:
                prefac = self.k0 / (self.pot_max - self.pot_min)
                self.gamd_pot = 0.5 * prefac * np.power(self.E - epot, 2)
                bias_force -= prefac * (self.E - epot) * self.gamd_forces

            if md_state.step < self.gamd_equil_steps:
                self._update_pot_distribution(epot)
                self._calc_E_k0()

            else:
                # free energy reweighting in production
                if self._check_boundaries(xi):

                    bink = self.get_index(xi)
                    self.histogram[bink[1], bink[0]] += 1

                    # first and second order cumulants for free energy reweighting
                    (
                        self.gamd_c1[bink[1], bink[0]],
                        self.gamd_m2[bink[1], bink[0]],
                        self.gamd_c2[bink[1], bink[0]],
                    ) = welford_var(
                        self.histogram[bink[1], bink[0]],
                        self.gamd_c1[bink[1], bink[0]],
                        self.gamd_m2[bink[1], bink[0]],
                        self.gamd_pot,
                    )

        if self.confine:
            bias_force += self.harmonic_walls(xi, delta_xi)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)
        self.gamd_pot_traj.append(self.gamd_pot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        if md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf,
                    "gamd_corr": self.gamd_corr,
                }
                self.write_output(output, filename="gamd.out")
                self.write_restart()

        return bias_force

    def get_pmf(self):

        kBT = kB_in_atomic * self.equil_temp
        self.gamd_corr = -self.gamd_c1 - self.gamd_c2 / (2.0 * kBT)

        self.pmf = -kBT * np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(self.histogram != 0),
        )
        self.pmf += self.gamd_corr
        self.pmf *= atomic_to_kJmol
        self.pmf -= self.pmf.min()

    def shared_bias(self):
        """TODO: fix me"""
        pass

    def _update_pot_distribution(self, epot: float):
        """update min, max, avg, var and std of epot

        Args:
            epot: potential energy
        """
        self.pot_min = np.min([epot, self.pot_min])
        self.pot_max = np.max([epot, self.pot_max])
        self.pot_count += 1
        self.pot_avg, self.pot_m2, self.pot_var = welford_var(
            self.pot_count, self.pot_avg, self.pot_m2, epot
        )
        self.pot_std = np.sqrt(self.pot_var)

    def _calc_E_k0(self):
        """compute force constant for gamd boost potential

        Args:
            epot: potential energy
        """
        if self.gamd_bound.lower() == "lower":
            self.E = self.pot_max
            ko = (self.sigma0 / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )
            self.k0 = np.min([1.0, ko])

        elif self.gamd_bound.lower() == "upper":
            ko = (1.0 - self.sigma0 / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )
            if 0.0 < ko <= 1.0:
                self.k0 = ko
            else:
                self.k0 = 1.0
            self.E = self.pot_min + (self.pot_max - self.pot_min) / self.k0

        elif self.gamd_bound.lower() == "amd":
            self.E = self.pot_max
 
        else:
            raise ValueError(f" >>> Error: unknown GaMD bound {self.gamd_bound}!")

    def write_restart(self, filename: str = "restart_gamd"):
        """write restart file
        TODO: fix me

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            pmf=self.pmf,
            c1=self.gamd_c1,
            m2=self.gamd_m2,
            corr=self.gamd_corr,
            pot_count=self.pot_count,
            pot_var=self.pot_var,
            pot_std=self.pot_std,
            pot_m2=self.pot_m2,
            pot_avg=self.pot_avg,
            pot_min=self.pot_min,
            pot_max=self.pot_max,
            k0=self.k0,
        )

    def restart(self, filename: str = "restart_gamd"):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.pmf = data["pmf"]
        self.gamd_c1 = data["c1"]
        self.gamd_m2 = data["m2"]
        self.gamd_corr = data["corr"]
        self.pot_count = data["pot_count"]
        self.pot_var = data["pot_var"]
        self.pot_std = data["pot_std"]
        self.pot_m2 = data["pot_m2"]
        self.pot_avg = data["pot_avg"]
        self.pot_min = data["pot_min"]
        self.pot_max = data["pot_max"]
        self.k0 = data["k0"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def write_traj(self):
        """save trajectory for post-processing"""
        data = {
            "E gamd [H]": self.gamd_pot_traj,
            "E pot [H]": self.epot,
            "T [K]": self.temp,
        }
        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.gamd_pot_traj = [self.gamd_pot_traj[-1]]
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
