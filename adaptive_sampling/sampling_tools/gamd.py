import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var
from ..processing_tools.thermodynamic_integration import integrate


class GaMD(EnhancedSampling):
    def __init__(
        self,
        sigma0: float,
        gamd_init_steps: int,
        gamd_equil_steps: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.sigma0 = sigma0
        self.gamd_init_steps = gamd_init_steps
        self.gamd_equil_steps = gamd_equil_steps

        self.pot_count = 0
        self.pot_var = 0.0
        self.pot_m2 = 0.0
        self.pot_mean = 0.0
        self.pot_min = 0.0
        self.pot_max = 0.0
        self.kg = 0.0

        self.gamd_pot = 0.0
        self.gamd_pot_traj = []

        self.c1 = np.zeros_like(self.histogram)
        self.c2 = np.zeros_like(self.histogram)
        self.m2 = np.zeros_like(self.histogram)
        self.corr = np.zeros_like(self.histogram)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):
        # TODO: fix me

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # get energy and forces
        bias_force = np.zeros_like(md_state.forces)
        self.gamd_forces = np.copy(md_state.forces)
        epot = np.copy(md_state.epot)
        self._get_force_constant(epot)

        # apply boost potential
        if md_state.step > self.gamd_init_steps:
            if epot < self.pot_max:
                prefac = self.kg / (2.0 * (self.pot_max - self.pot_min))
                dV = prefac * np.power(self.pot_max - self.gamd_pot, 2)
                self.gamd_pot += dV
                bias_force -= (
                    2.0 * prefac * (self.pot_max - self.gamd_pot) * self.gamd_forces
                )
            else:
                self.gamd_pot = 0.0

        # first and second order cumulants for free energy reweighting
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            # first and second order cumulants for free energy reweighting
            mean, m2, var = welford_var(
                self.histogram[bink[1], bink[0]],
                self.c1[bink[1], bink[0]],
                self.m2[bink[1], bink[0]],
                self.gamd_pot,
            )

            self.c1[bink[1], bink[0]] = mean
            self.m2[bink[1], bink[0]] = m2
            self.c2[bink[1], bink[0]] = var

        else:
            bias_force += self.harmonic_walls(xi, delta_xi)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)
        self.gamd_pot_traj.append(self.gamd_pot)

        # correction for kinetics
        if self.kinetics:
            self.kinetics(delta_xi)

        if self.the_md.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf,
                    "gamd_pot_c1": self.c1,
                    "gamd_pot_c2": self.c2,
                }
                self.write_output(output, filename="gamd.out")
                self.write_restart()

        return bias_force

    def get_pmf(self):

        kB_a = 1.380648e-23 / 4.359744e-18
        kBT = kB_a * self.equil_temp
        self.corr = -self.c1 - self.c2 / (2.0 * kBT)

        self.pmf = -kBT * np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(self.histogram != 0),
        )
        self.pmf += self.corr
        self.pmf *= 2625.499639  # Hartree to kJ/mol
        self.pmf -= self.pmf.min()

    def shared_bias(self):
        """TODO: fix me"""
        pass

    def _get_force_constant(self, epot):
        """compute force constant for gamd boost potential"""
        # compute gamd boost
        if self.the_md.step <= self.gamd_equil_steps:
            self.pot_count += 1
            self.pot_mean, self.pot_m2, self.pot_var = welford_var(
                self.pot_count, self.pot_mean, self.pot_m2, epot
            )
            self.pot_min = np.min([epot, self.pot_min])
            self.pot_max = np.max([epot, self.pot_max])

            # unsing lower bound for force constant
            if self.pot_var != 0.0:
                k0 = (self.sigma0 / np.sqrt(self.pot_var)) * (
                    (self.pot_max - self.pot_min) / (self.pot_max - self.pot_mean)
                )
                self.kg = np.min([1.0, k0])

    def write_restart(self, filename: str = "restart_gamd"):
        """write restart file
        TODO: fix me

        args:
            filename: name of restart file
        """
        pass

    def restart(self, filename: str = "restart_gamd"):
        """TODO: fix me"""
        pass

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
