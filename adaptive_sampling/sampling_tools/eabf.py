import random
import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, diff, cond_avg
from ..processing_tools.thermodynamic_integration import integrate


class eABF(EnhancedSampling):
    def __init__(
        self,
        ext_sigma: list,
        ext_mass: list,
        *args,
        nfull: int = 100,
        friction: float = 1.0e-3,
        seed_in: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        (xi, _) = self.get_cv()

        # for dynamics of extended-system
        kB_a = 1.380648e-23 / 4.359744e-18
        self.ext_sigma = self.unit_conversion_cv(np.asarray(ext_sigma))[0]
        self.ext_k = (kB_a * self.equil_temp) / (self.ext_sigma * self.ext_sigma)
        self.ext_mass = np.asarray(ext_mass)
        self.ext_hist = np.zeros_like(self.histogram)
        self.ext_forces = np.zeros(self.ncoords)
        self.ext_momenta = np.zeros(self.ncoords)
        self.ext_coords = np.copy(xi)
        self.correction_czar = np.zeros_like(self.bias)
        self.czar_force = np.zeros_like(self.bias)
        self.friction = friction
        self.ext_traj = np.copy(self.traj)

        # for ABF
        self.nfull = nfull
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)

        # set random seed for langevin dynamics
        if type(seed_in) is int:
            random.seed(seed_in)
        else:
            try:
                random.setstate(seed_in)
            except:
                if self.verbose:
                    print(
                        "\n >>> Warning: The provided seed was neither an int nor a state of random!\n"
                    )

        # initialize extended system at target temp of MD simulation
        au2k = 315775.04e0
        for i in range(self.ncoords):
            self.ext_momenta[i] = random.gauss(0.0, 1.0) * np.sqrt(
                self.equil_temp * self.ext_mass[i]
            )
            ttt = (np.power(self.ext_momenta, 2) / self.ext_mass).sum()
            ttt /= self.ncoords
            self.ext_momenta *= np.sqrt(self.equil_temp / (ttt * au2k))

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)
        self._propagate()

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        bias_force = np.zeros_like(md_state.forces)

        if (self.ext_coords <= self.maxx).all() and (
            self.ext_coords >= self.minx
        ).all():

            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink[1], bink[0]] += 1

            bias_force += self._extended_dynamics(xi, delta_xi, self.ext_sigma)

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.ext_hist[bink[1], bink[0]] > self.nfull
                    else self.ext_hist[bink[1], bink[0]] / self.nfull
                )

                # apply bias force on extended system
                (
                    self.bias[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    self.var_force[i][bink[1], bink[0]],
                ) = welford_var(
                    self.ext_hist[bink[1], bink[0]],
                    self.bias[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    self.ext_k[i] * diff(self.ext_coords[i], xi[i], self.cv_type[i]),
                )
                self.ext_forces -= ramp * self.bias[i][bink[1], bink[0]]

        else:
            bias_force += self._extended_dynamics(xi, delta_xi, self.ext_sigma)

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
            self.kinetics(delta_xi)

        if md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"bias force {i}"] = self.bias[i]
                    output[f"var force {i}"] = self.var_force[i]
                    output[f"czar force {i}"] = self.czar_force[i]

                self.write_output(output, filename="eabf.out")
                self.write_restart()

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):

        kB_a = 1.380648e-23 / 4.359744e-18

        log_rho = np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(0 != self.histogram),
        )
        avg_force = cond_avg(self.correction_czar, self.histogram)

        if self.ncoords == 1:
            self.czar_force[0] = (
                -kB_a * self.equil_temp * np.gradient(log_rho[0], self.grid[0])
                + avg_force[0]
            )
            self.pmf[0, :], _ = integrate(
                self.czar_force[0][0],
                self.dx,
                equil_temp=self.equil_temp,
                method=method,
            )
            self.pmf *= 2625.499639  # Hartree to kJ/mol

        else:
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
            self.czar_force[0] = -kB_a * self.equil_temp * der_log_rho[1] + avg_force[0]
            self.czar_force[1] = -kB_a * self.equil_temp * der_log_rho[0] + avg_force[1]
            if self.verbose:
                print(
                    " >>> Info: On-the-fly integration only available for 1D coordinates"
                )

    def shared_bias(self):
        """TODO"""
        pass

    def _propagate(self, langevin: bool = True):
        """Propagate momenta/coords of extended variable in time with Velocity Verlet

        Args:
           langevin: Temperature control with langevin dynamics
        """
        kB_a = 1.380648e-23 / 4.359744e-18
        if langevin:
            prefac = 2.0 / (2.0 + self.friction * self.the_md.dt)
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_a / 2.0e0
            )
            self.ext_rand_gauss = np.zeros(shape=(len(self.ext_momenta),), dtype=float)
            for atom in range(len(self.ext_rand_gauss)):
                self.ext_rand_gauss[atom] = random.gauss(0, 1)

            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords += (
                prefac * self.the_md.dt * self.ext_momenta / self.ext_mass
            )

        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords += self.the_md.dt * self.ext_momenta / self.ext_mass

    def _up_momenta(self, langevin: bool = True):
        """Update momenta of extended variables with Velocity Verlet

        Args:
            langevin: Temperature control with langevin dynamics
        """
        kB_a = 1.380648e-23 / 4.359744e-18
        if langevin:
            prefac = (2.0e0 - self.friction * self.the_md.dt) / (
                2.0e0 + self.friction * self.the_md.dt
            )
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_a / 2.0e0
            )
            self.ext_momenta *= prefac
            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces

    def _extended_dynamics(
        self, xi: np.ndarray, delta_xi: np.ndarray, margin: np.ndarray
    ) -> np.ndarray:
        """get forces due to coupling to extended system and confine extended variable to range of interest

        args:
            margin: inset from minx, maxx where bias starts

        returns:
            bias_force:
        """
        bias_force = np.zeros_like(self.the_md.forces)

        for i in range(self.ncoords):
            # harmonic coupling of extended coordinate to reaction coordinate

            dxi = diff(self.ext_coords[i], xi[i], self.cv_type[i])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_force[i] -= self.ext_k[i] * dxi
            bias_force += bias_force[i] * delta_xi[i]

            if self.ext_coords[i] > (self.maxx[i] - margin[i]):
                r = diff(self.maxx[i] - margin[i], self.ext_coords[i], self.cv_type[i])
                self.ext_forces[i] -= self.f_conf[i] * r

            elif self.ext_coords[i] < (self.minx[i] + margin[i]):
                r = diff(self.minx[i] + margin[i], self.ext_coords[i], self.cv_type[i])
                self.ext_forces[i] -= self.f_conf[i] * r

        return bias_force

    def write_restart(self, filename: str = "restart_abf"):
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
        )

    def restart(self, filename: str = "restart_abf"):
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

    def write_traj(self):
        """save trajectory for post-processing"""

        data = {}
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.ext_traj[:, i] /= np.pi / 180.0
            elif self.cv_type[i] == "distance":
                self.ext_traj[:, i] *= 0.52917721092e0
            data[f"lambda{i}"] = self.ext_traj[:, i]
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
