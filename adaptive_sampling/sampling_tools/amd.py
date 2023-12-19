import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var
from ..units import *


class aMD(EnhancedSampling):
    """Accelerated Molecular Dynamics
    
        see:
            aMD: Hamelberg et. al., J. Chem. Phys. 120, 11919 (2004); https://doi.org/10.1063/1.1755656
            GaMD: Miao et. al., J. Chem. Theory Comput. (2015); https://doi.org/10.1021/acs.jctc.5b00436
            SaMD: Zhao et. al., J. Phys. Chem. Lett. 14, 4, 1103 - 1112 (2023); https://doi.org/10.1021/acs.jpclett.2c03688

        Apply global boost potential to potential energy, that is independent of Collective Variables.

    Args:
        amd_parameter: acceleration parameter; SaMD, GaMD == sigma0; aMD == alpha
        init_step: initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps: equilibration steps, min, max and var of potential energy is still updated
                          force constant of coupling is calculated from previous steps
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        amd_method: "aMD": apply accelerated MD
                    "GaMD_lower": use lower bound for GaMD boost (default)
                    "GaMD_upper: use upper bound for GaMD boost
                    "SaMD: apply Sigmoid accelerated MD
        confine: if system should be confined to range of CV
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
        samd_c0: c0 constant for SaMD 
    """

    def __init__(
        self,
        amd_parameter: float,
        init_steps: int,
        equil_steps: int,
        *args,
        amd_method: str = "gamd_lower",
        confine: bool = True,
        samd_c0: float = 0.0001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.parameter = amd_parameter
        self.init_steps = init_steps
        self.equil_steps = equil_steps
        self.amd_method = amd_method.lower()
        self.confine = confine

        if self.verbose and amd_method.lower() == "amd":
            print(f" >>> Warning: Please use GaMD or SaMD to obtain accurate free energy estimates!\n")

        self.pot_count = 0
        self.pot_var = 0.0
        self.pot_std = 0.0
        self.pot_m2 = 0.0
        self.pot_avg = 0.0
        self.pot_min = +np.inf
        self.pot_max = -np.inf
        self.k0 = 0.0
        self.k1 = 0.0
        self.k = 0.0
        self.E = 0.0
        self.c0 = samd_c0
        self.c = 1/self.c0 - 1

        self.amd_pot = 0.0
        self.amd_pot_traj = []

        self.amd_c1 = np.zeros_like(self.histogram)
        self.amd_c2 = np.zeros_like(self.histogram)
        self.amd_m2 = np.zeros_like(self.histogram)
        self.amd_corr = np.zeros_like(self.histogram)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        # get energy and forces
        bias_force = np.zeros_like(md_state.forces)
        self.amd_forces = np.copy(md_state.forces)
        epot = md_state.epot

        if md_state.step < self.init_steps:
            self._update_pot_distribution(epot)

        else:
            if md_state.step == self.init_steps:
                self._calc_E_k0()

            # apply boost potential
            bias_force += self._apply_boost(epot)

            if md_state.step < self.equil_steps:
                self._update_pot_distribution(epot)
                self._calc_E_k0()

            else:
                # free energy reweighting in production
                if self._check_boundaries(xi):

                    bink = self.get_index(xi)
                    self.histogram[bink[1], bink[0]] += 1

                    # first and second order cumulants for free energy reweighting
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

        if self.confine:
            bias_force += self.harmonic_walls(xi, delta_xi)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)
        self.amd_pot_traj.append(self.amd_pot)

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
                    "amd_corr": self.amd_corr,
                }
                self.write_output(output, filename="amd.out")
                self.write_restart()

        return bias_force

    def get_pmf(self):

        kBT = kB_in_atomic * self.equil_temp
        self.amd_corr = -self.amd_c1 - self.amd_c2 / (2.0 * kBT)

        self.pmf = -kBT * np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(self.histogram != 0),
        )
        self.pmf += self.amd_corr
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
        """compute force constant for amd boost potential

        Args:
            epot: potential energy
        """
        if self.amd_method.lower() == "gamd_lower":
            self.E = self.pot_max
            ko = (self.parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )

            self.k0 = np.min([1.0, ko])

        elif self.amd_method.lower() == "gamd_upper":
            ko = (1.0 - self.parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_avg - self.pot_min)
            )
            if 0.0 < ko <= 1.0:
                self.k0 = ko
            else:
                self.k0 = 1.0
            self.E = self.pot_min + (self.pot_max - self.pot_min) / self.k0

        elif self.amd_method.lower() == "samd":
            ko = (self.parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )

            self.k0 = np.min([1.0, ko])
            if (self.pot_std / self.parameter) <= 1.0:
                self.k = self.k0
            else:
                self.k1 = np.max([0,(np.log(self.c) + np.log((self.pot_std)/(self.parameter) - 1))/(self.pot_avg - self.pot_min)])
                self.k = np.max([self.k0,self.k1])

        elif self.amd_method.lower() == "amd":
            self.E = self.pot_max
 
        else:
            raise ValueError(f" >>> Error: unknown aMD method {self.amd_method}!")

    def _apply_boost(self, epot):
        """Apply boost potential to forces
        """
        if self.amd_method.lower() not in ["amd", "gamd_lower", "gamd_upper", "samd"]:
            raise ValueError(f" >>> Error: unknown aMD method {self.amd_method}!")

        if epot < self.E:
            if self.amd_method.lower() == "amd":
                self.amd_pot = np.square(self.E - epot) / (self.parameter + (self.E - epot))
                boost_force = -(
                    ((epot - self.E) * (epot - 2.0 * self.parameter - self.E)) / np.square(epot - self.parameter - self.E)
                ) * self.amd_forces

            elif self.amd_method.lower() in ["gamd_lower", "gamd_upper"]:
                prefac = self.k0 / (self.pot_max - self.pot_min)
                self.amd_pot = 0.5 * prefac * np.power(self.E - epot, 2)
                boost_force = -prefac * (self.E - epot) * self.amd_forces
        else:
            boost_force = 0.0
            self.amd_pot = 0.0

        if self.amd_method.lower() == "samd":
            self.amd_pot = self.pot_max - epot - 1/self.k * np.log((self.c + np.exp(self.k * (self.pot_max - self.pot_min))) 
               / (self.c + np.exp(self.k * (epot - self.pot_min))))
            boost_force = (1.0/(np.exp(-self.k * (epot - self.pot_min) + np.log((1/self.c0) - 1)) + 1) - 1) * self.amd_forces

        return boost_force
    
    def write_restart(self, filename: str = "restart_amd"):
        """write restart file
        TODO: fix me

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            pmf=self.pmf,
            c1=self.amd_c1,
            m2=self.amd_m2,
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

    def restart(self, filename: str = "restart_amd"):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.pmf       = data["pmf"]
        self.amd_c1    = data["c1"]
        self.amd_m2    = data["m2"]
        self.amd_corr  = data["corr"]
        self.pot_count = data["pot_count"]
        self.pot_var   = data["pot_var"]
        self.pot_std   = data["pot_std"]
        self.pot_m2    = data["pot_m2"]
        self.pot_avg   = data["pot_avg"]
        self.pot_min   = data["pot_min"]
        self.pot_max   = data["pot_max"]
        self.k0        = data["k0"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def write_traj(self):
        """save trajectory for post-processing"""
        data = {
            "E amd [H]": self.amd_pot_traj,
            "E pot [H]": self.epot,
            "T [K]": self.temp,
        }
        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.amd_pot_traj = [self.amd_pot_traj[-1]]
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
