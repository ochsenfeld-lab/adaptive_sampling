import numpy as np
from .enhanced_sampling import EnhancedSampling
from .metaeabf import WTMeABF
from .gamd import GaMD
from .utils import welford_var, diff, cond_avg
from ..processing_tools.thermodynamic_integration import integrate
from ..units import *


class GaWTMeABF(WTMeABF, GaMD, EnhancedSampling):

    def __init__(self, *args, do_wtm: bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_wtm = do_wtm

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        epot = md_state.epot
        self.gamd_forces = np.copy(md_state.forces)

        (xi, delta_xi) = self.get_cv(**kwargs)

        self._propagate()
        if self.do_wtm:
            bias_force = self._extended_dynamics(xi, delta_xi, self.hill_std)
        else:
            bias_force = self._extended_dynamics(xi, delta_xi)

        if md_state.step < self.gamd_init_steps:
            self._update_pot_distribution(epot)

        else:
            if md_state.step == self.gamd_init_steps:
                self._calc_E_k0()

            # apply gamd boost potential
            prefac = self.k0 / (self.pot_max - self.pot_min)
            self.gamd_pot = 0.5 * prefac * np.power(self.E - epot, 2)
            bias_force -= prefac * (self.E - epot) * self.gamd_forces

            if md_state.step < self.gamd_equil_steps:
                self._update_pot_distribution(epot)
                self._calc_E_k0()

            else:

                # (WTM-)eABF bias on extended-variable only in production
                if self.do_wtm:
                    mtd_forces = self.get_mtd_force(self.ext_coords)

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
                        self.ext_forces -= (
                            ramp * self.abf_forces[i][bink[1], bink[0]] 
                        )
                        
                        if self.do_wtm:
                            self.ext_forces -= self.metapot[i]

        # free energy reweighting
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            # CZAR
            for i in range(self.ncoords):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]], self.cv_type[i])
                self.correction_czar[i][bink[1], bink[0]] += self.ext_k[i] * dx

                # GaMD
                if md_state.step >= self.gamd_equil_steps:

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

        self._up_momenta()

        self.traj = np.append(self.traj, [xi], axis=0)
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.temp.append(md_state.temp)
        self.gamd_pot_traj.append(self.gamd_pot)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self.kinetics(delta_xi)

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
                    output[f"czar force {i}"] = self.czar_force[i]
                output[f"metapot"] = self.metapot
                output[f"GaMD corr"] = self.gamd_corr

                self.write_output(output, filename="metaeabf.out")
                self.write_restart()

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):

        log_rho = np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(0 != self.histogram),
        )
        avg_force = cond_avg(self.correction_czar, self.histogram)

        self.gamd_corr = -self.gamd_c1 - self.gamd_c2 / (
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
            self.pmf += self.gamd_corr
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

    def write_restart(self, filename: str = "restart_metaabf"):
        """write restart file

        args:
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
            gamd_c1=self.gamd_c1,
            gamd_m2=self.gamd_m2,
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

    def restart(self, filename: str = "restart_metaabf"):
        """restart from restart file

        args:
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

    def write_traj(self):
        """save trajectory for post-processing"""

        data = {}
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.ext_traj[:, i] /= np.pi / 180.0
            elif self.cv_type[i] == "distance":
                self.ext_traj[:, i] *= 0.52917721092e0
            data[f"lambda{i}"] = self.ext_traj[:, i]
        data[f"E_gamd [H]"] = self.gamd_pot_traj
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
