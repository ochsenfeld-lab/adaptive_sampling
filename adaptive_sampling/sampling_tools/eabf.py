import random, os, time
import numpy as np
from typing import Union
from .enhanced_sampling import EnhancedSampling
from .abf import ABF
from .utils import (
    diff,
    correct_periodicity,
    welford_var,
    combine_welford_stats,
    cond_avg,
)
from ..processing_tools.thermodynamic_integration import integrate


class eABF(ABF, EnhancedSampling):
    """Extended-system adaptive biasing Force method

       see: Lesage et. al., J. Phys. Chem. B (2017); https://doi.org/10.1021/acs.jpcb.6b10055

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particle is biased with the ABF algorithm.

    Args:
        md: Object of the MDInterface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
            [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        ext_sigma: thermal width of coupling between collective and extended variable
            if None, it will be estimated based on the standard deviation of the CV in an initial MD
        ext_mass: mass of extended variable in atomic units
        adaptive_coupling_stride: initial MD steps to estimate ext_sigma
        adaptive_coupling_scaling: scaling factor for standard deviation of initial MD to ext_sigma
        adaptive_coupling_min: minimum for ext_sigma from adaptive estimate
        nfull: Number of force samples per bin where full bias is applied,
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        friction: friction coefficient for Langevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        equil_temp: equilibrium temperature of MD
        verbose: print verbose information
        kinetics: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """

    def __init__(
        self,
        *args,
        ext_sigma: Union[float, list] = None,
        ext_mass: Union[float, list] = 20.0,
        adaptive_coupling_stride: int = 5000,
        adaptive_coupling_scaling: float = 0.5,
        adaptive_coupling_min: float = 0.01,
        friction: float = 1.0e-3,
        seed_in: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # extended system
        if not hasattr(ext_sigma, "__len__") and not ext_sigma:
            self.estimate_sigma = True
        elif hasattr(ext_sigma, "__len__") and any(std is None for std in ext_sigma):
            self.estimate_sigma = True
        else:
            self.estimate_sigma = False

        ext_sigma = (
            [ext_sigma for _ in range(self.ncoords)]
            if not hasattr(ext_sigma, "__len__")
            else ext_sigma
        )
        ext_mass = (
            [ext_mass for _ in range(self.ncoords)]
            if not hasattr(ext_mass, "__len__")
            else ext_mass
        )

        (xi, _) = self.get_cv()

        self.ext_mass = np.asarray(ext_mass)
        self.ext_hist = np.zeros_like(self.histogram)
        self.ext_forces = np.zeros(self.ncoords)
        self.ext_momenta = np.zeros(self.ncoords)
        self.ext_coords = np.copy(xi)
        self.correction_czar = np.zeros_like(self.bias)
        self.czar_force = np.zeros_like(self.bias)
        self.friction = friction
        self.ext_traj = np.copy(self.traj)

        self.adaptive_coupling_stride = adaptive_coupling_stride
        self.adaptive_coupling_counter = 0
        self.adaptive_coupling_m2 = np.zeros(self.ncoords)
        self.adaptive_coupling_mean = np.zeros(self.ncoords)
        self.adaptive_coupling_var = np.zeros(self.ncoords)
        self.adaptive_coupling_scaling = adaptive_coupling_scaling
        self.adaptive_coupling_min = np.full(self.ncoords, adaptive_coupling_min)

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
        # for dynamics of extended-system
        if not self.estimate_sigma:
            self.ext_sigma = self.unit_conversion_cv(np.asarray(ext_sigma))[0]
            self.ext_k = (kB_in_atomic * self.equil_temp) / (
                self.ext_sigma * self.ext_sigma
            )
            self.reinit_ext_system(xi)

        if self.verbose:
            print(" >>> INFO: Extended-system Parameters:")
            print("\t ---------------------------------------------")
            print(
                f"\t Coupling:\t{self.ext_sigma if not self.estimate_sigma else f'estimate from {self.adaptive_coupling_stride} steps'}"
            )
            print(f"\t Masses:\t{self.ext_mass}")
            print("\t ---------------------------------------------")

    def step_bias(
        self,
        output_file: str = "eabf.out",
        traj_file: str = "CV_traj.dat",
        restart_file: str = "restart_eabf",
        stabilize: bool = False,
        stabilizer_threshold: float = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply eABF to MD simulation

         Args:
             write_output: if on-the-fly free energy estimate and restart files should be written
             write_traj: if CV and extended system trajectory file should be written
             stabilize: if stabilisation algorithm should be applied for discontinous CVs
             stabilizer_threshold: treshold for stabilisation of extended system

        Returns:
             bias_force: Adaptive biasing force of current step that has to be added to molecular forces
        """
        from ..units import kB_in_atomic
        self.md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        # obtain coupling strength from initial MD
        if self.estimate_sigma and self.md_state.step < self.adaptive_coupling_stride:
            self.ext_sigma = self.estimate_coupling(xi) * self.adaptive_coupling_scaling
            return np.zeros_like(self.md_state.coords)

        elif (
            self.estimate_sigma and self.md_state.step == self.adaptive_coupling_stride
        ):
            self.ext_sigma = self.estimate_coupling(xi) * self.adaptive_coupling_scaling
            for i, s in enumerate(self.ext_sigma):
                if s < self.adaptive_coupling_min[i]:
                    print(
                        f" >>> WARNING: estimated coupling of extended-system is suspiciously small ({s}). Resetting to {self.adaptive_coupling_min[i]}."
                    )
                    self.ext_sigma[i] = self.adaptive_coupling_min[i]
            if self.verbose:
                print(
                    f" >>> INFO: setting coupling width of extended-system to {self.ext_sigma}!"
                )
            self.ext_k = (kB_in_atomic * self.equil_temp) / (
                self.ext_sigma * self.ext_sigma
            )

            with open("COUPLING", "w") as out:
                for s in self.ext_sigma:
                    out.write(f"{s}\t")

            self.reinit_ext_system(xi)

        if stabilize and len(self.traj) > 0:
            self.stabilizer(xi, threshold=stabilizer_threshold)

        self._propagate()

        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.ext_sigma)
        force_sample = [0 for _ in range(2 * self.ncoords)]

        if self._check_boundaries(self.ext_coords):

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.ext_hist[bin_la[1], bin_la[0]] > self.nfull
                    else self.ext_hist[bin_la[1], bin_la[0]] / self.nfull
                )

                # apply bias force on extended system
                force_sample[i] = self.ext_k[i] * diff(
                    self.ext_coords[i], xi[i], self.periodicity[i]
                )
                (
                    self.bias[i][bin_la[1], bin_la[0]],
                    self.m2_force[i][bin_la[1], bin_la[0]],
                    self.var_force[i][bin_la[1], bin_la[0]],
                ) = welford_var(
                    self.ext_hist[bin_la[1], bin_la[0]],
                    self.bias[i][bin_la[1], bin_la[0]],
                    self.m2_force[i][bin_la[1], bin_la[0]],
                    force_sample[i],
                )
                self.ext_forces -= ramp * self.bias[i][bin_la[1], bin_la[0]]

        # xi-conditioned accumulators for CZAR
        if self._check_boundaries(xi):

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                force_sample[self.ncoords + i] = self.ext_k[i] * diff(
                    self.ext_coords[i], self.grid[i][bink[i]], self.periodicity[i]
                )
                self.correction_czar[i][bink[1], bink[0]] += force_sample[
                    self.ncoords + i
                ]

        # shared-bias eABF
        if self.shared:
            self.shared_bias(
                xi,
                force_sample,
                **kwargs,
            )

        if traj_file:
            self.traj = np.append(self.traj, [xi], axis=0)
            self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
            self.temp.append(self.md_state.temp)
            self.epot.append(self.md_state.epot)

        self._up_momenta()

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        if self.md_state.step % self.out_freq == 0:
            # write output

            if traj_file and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)

            if output_file:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"bias force {i}"] = self.bias[i]
                    output[f"var force {i}"] = self.var_force[i]
                    output[f"czar force {i}"] = self.czar_force[i]

                self.write_output(output, filename=output_file)

            if restart_file:
                self.write_restart(filename=restart_file)

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):
        from ..units import kB_in_atomic, atomic_to_kJmol
        log_rho = np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(0 != self.histogram),
        )
        avg_force = cond_avg(self.correction_czar, self.histogram)

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
            if self.verbose and self.the_md.get_sampling_data().step == 1:
                print(
                    " >>> Info: On-the-fly integration only available for 1D coordinates"
                )
        return self.pmf

    def shared_bias(
        self,
        xi,
        force_sample,
        sync_interval: int = 50,
        mw_file: str = "../shared_bias",
        n_trials: int = 10,
    ):
        """syncs eABF bias with other walkers

        TODO: 2D collective variables

        Args:
            xi: state of the collective variable
            force_sample: force sample of current step
            sync_interval: number of steps between sychronisation
            mw_file: name of buffer file for shared-bias
            n_trials: number of attempts to access of buffer file before throwing an error
        """
        md_state = self.the_md.get_sampling_data()
        if md_state.step == 0:
            if self.verbose:
                print(" >>> Info: Creating a new instance for shared-bias eABF.")
                print(
                    " >>> Info: Data of local walker stored in `restart_eabf_local.npz`."
                )

            # create seperate restart file with local data only
            self._write_restart(
                filename="restart_eabf_local",
                hist=self.histogram,
                force=self.bias,
                var=self.var_force,
                m2=self.m2_force,
                ext_hist=self.ext_hist,
                czar_corr=self.correction_czar,
            )
            self.update_samples = np.zeros(shape=(sync_interval, len(force_sample)))
            self.last_samples_xi = np.zeros(shape=(sync_interval, len(xi)))
            self.last_samples_la = np.zeros(shape=(sync_interval, len(self.ext_coords)))

            if not os.path.isfile(mw_file + ".npz"):
                if self.verbose:
                    print(
                        f" >>> Info: Creating buffer file for shared-bias eABF: `{mw_file}.npz`."
                    )
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    force=self.bias,
                    var=self.var_force,
                    m2=self.m2_force,
                    ext_hist=self.ext_hist,
                    czar_corr=self.correction_czar,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(
                    f" >>> Info: Syncing with existing buffer file for shared-bias eABF: `{mw_file}.npz`."
                )

        count = md_state.step % sync_interval
        self.update_samples[count] = force_sample
        self.last_samples_xi[count] = xi
        self.last_samples_la[count] = self.ext_coords

        if count == sync_interval - 1:

            # calculate progress since last sync from new samples
            hist = np.zeros_like(self.histogram)
            m2 = np.zeros_like(self.m2_force)
            bias = np.zeros_like(self.bias)
            ext_hist = np.zeros_like(self.ext_hist)
            czar_corr = np.zeros_like(self.correction_czar)

            for i, sample in enumerate(self.update_samples):
                if self._check_boundaries(self.last_samples_la[i]):
                    bin_la = self.get_index(self.last_samples_la[i])
                    ext_hist[bin_la[1], bin_la[0]] += 1
                    for j in range(self.ncoords):
                        (
                            bias[j][bin_la[1], bin_la[0]],
                            m2[j][bin_la[1], bin_la[0]],
                            _,
                        ) = welford_var(
                            ext_hist[bin_la[1], bin_la[0]],
                            bias[j][bin_la[1], bin_la[0]],
                            m2[j][bin_la[1], bin_la[0]],
                            sample[j],
                        )

                if self._check_boundaries(self.last_samples_xi[i]):
                    bin_xi = self.get_index(self.last_samples_xi[i])
                    hist[bin_xi[1], bin_xi[0]] += 1
                    for j in range(self.ncoords):
                        czar_corr[j][bin_xi[1], bin_xi[0]] += sample[self.ncoords + j]

            self._update_eabf(
                "restart_eabf_local",
                hist,
                ext_hist,
                bias,
                m2,
                czar_corr,
            )

            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):

                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666)
                    self._update_eabf(
                        mw_file,
                        hist,
                        ext_hist,
                        bias,
                        m2,
                        czar_corr,
                    )
                    self.restart(filename=mw_file, restart_ext_sys=False)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again

                    self.get_pmf()  # get new global pmf
                    break

                elif trial < n_trials:
                    if self.verbose:
                        print(
                            f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts."
                        )
                    time.sleep(0.1)
                else:
                    raise Exception(
                        f" >>> Fatal Error: Failed to sync bias with `{mw_file}.npz`."
                    )

    def reinit_ext_system(self, xi: np.ndarray):
        """Initialize extended-system dynamics with random momenta"""
        from ..units import atomic_to_K
        self.ext_coords = np.copy(xi)
        for i in range(self.ncoords):
            self.ext_momenta[i] = random.gauss(0.0, 1.0) * np.sqrt(
                self.equil_temp * self.ext_mass[i]
            )
            ttt = (np.power(self.ext_momenta, 2) / self.ext_mass).sum()
            ttt /= self.ncoords
            self.ext_momenta *= np.sqrt(self.equil_temp / (ttt * atomic_to_K))

    def stabilizer(self, xi: np.ndarray, threshold: float = None):
        """Stabilize extended dynamics in case of discontiouity in Collective Variable

        see: Hulm et al., J. Chem. Theory Comput. (2023)

        Args:
            xi: current value of the CV
            threshold: treshold for discontinuity in the CV (default: ext_sigma)
        """
        if threshold == None:
            threshold = self.ext_sigma

        for i in range(self.ncoords):
            if abs(xi[i] - self.traj[-1][i]) > threshold[i]:
                delta = diff(self.ext_coords[i], self.traj[-1][i], self.periodicity[i])
                self.ext_coords[i] = xi[i] + delta
                if self.verbose:
                    print(
                        f" >>> INFO: extended system corrected after discontinuity of CV{i}!"
                    )

    def _propagate(self, langevin: bool = True):
        """Propagate momenta/coords of extended variable in time with Velocity Verlet

        Args:
           langevin: Temperature control with langevin dynamics
        """
        from ..units import kB_in_atomic
        if langevin:
            prefac = 2.0 / (2.0 + self.friction * self.the_md.dt)
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_in_atomic / 2.0e0
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

        for i in range(self.ncoords):
            self.ext_coords[i] = correct_periodicity(
                self.ext_coords[i], self.periodicity[i]
            )

    def _up_momenta(self, langevin: bool = True):
        """Update momenta of extended variables with Velocity Verlet

        Args:
            langevin: Temperature control with langevin dynamics
        """
        from ..units import kB_in_atomic
        if langevin:
            prefac = (2.0e0 - self.friction * self.the_md.dt) / (
                2.0e0 + self.friction * self.the_md.dt
            )
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_in_atomic / 2.0e0
            )
            self.ext_momenta *= prefac
            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces

    def _extended_dynamics(
        self,
        xi: np.ndarray,
        delta_xi: np.ndarray,
        margin: np.ndarray = np.array([0.0, 0.0]),
    ) -> np.ndarray:
        """get forces due to coupling to extended system and confine extended variable to range of interest

        Args:
            xi: collective variable
            delta_xi: gradient of collective variable
            margin: inset from minx, maxx where bias starts

        Returns:
            bias_force:
        """
        bias_force = np.zeros_like(self.the_md.forces)

        for i in range(self.ncoords):
            # harmonic coupling of extended coordinate to reaction coordinate
            dxi = diff(self.ext_coords[i], xi[i], self.periodicity[i])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_force -= self.ext_k[i] * dxi * delta_xi[i]

            # harmonic walls for confinement to range of interest
            if self.f_conf[i] > 0.0:
                if (
                    self.ext_coords[i] > (self.maxx[i] - margin[i])
                    and not self.periodicity[i]
                ):
                    r = diff(
                        self.maxx[i] - margin[i],
                        self.ext_coords[i],
                        self.periodicity[i],
                    )
                    self.ext_forces[i] -= self.f_conf[i] * r

                elif (
                    self.ext_coords[i] < (self.minx[i] + margin[i])
                    and not self.periodicity[i]
                ):
                    r = diff(
                        self.minx[i] + margin[i],
                        self.ext_coords[i],
                        self.periodicity[i],
                    )
                    self.ext_forces[i] -= self.f_conf[i] * r

        return bias_force

    def estimate_coupling(self, cv):
        """Adaptive estimate of coupling width from trajectory"""
        self.adaptive_coupling_counter += 1
        tau = self.adaptive_coupling_stride
        if self.adaptive_coupling_counter < self.adaptive_coupling_stride:
            tau = self.adaptive_coupling_counter
        (
            self.adaptive_coupling_mean,
            self.adaptive_coupling_m2,
            self.adaptive_coupling_var,
        ) = welford_var(
            self.md_state.step,
            self.adaptive_coupling_mean,
            self.adaptive_coupling_m2,
            cv,
            tau,
        )
        return np.sqrt(self.adaptive_coupling_var)

    def _update_eabf(
        self,
        filename: str,
        hist,
        ext_hist,
        bias,
        m2,
        czar_corr,
    ):
        with np.load(f"{filename}.npz") as data:

            new_hist = data["ext_hist"] + hist
            new_czar_corr = data["czar_corr"] + czar_corr
            new_ext_hist = np.zeros_like(self.histogram).flatten()
            new_m2 = np.zeros_like(self.m2_force).flatten()
            new_bias = np.zeros_like(self.bias).flatten()

            for i in range(len(new_ext_hist)):
                (new_ext_hist[i], new_bias[i], new_m2[i], _,) = combine_welford_stats(
                    data["ext_hist"].flatten()[i],
                    data["force"].flatten()[i],
                    data["m2"].flatten()[i],
                    ext_hist.flatten()[i],
                    bias.flatten()[i],
                    m2.flatten()[i],
                )

        self._write_restart(
            filename=filename,
            hist=new_hist,
            force=new_bias.reshape(self.bias.shape),
            m2=new_m2.reshape(self.m2_force.shape),
            ext_hist=new_ext_hist.reshape(self.ext_hist.shape),
            czar_corr=new_czar_corr,
        )

    def write_restart(self, filename: str = "restart_abf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            force=self.bias,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
        )

    def restart(self, filename: str = "restart_abf", restart_ext_sys=False):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz")
        except:
            raise OSError(f" >>> Fatal Error: restart file `{filename}.npz` not found!")

        self.histogram = data["hist"]
        self.bias = data["force"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords = data["ext_coords"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def _write_ext_traj(self):
        from ..units import BOHR_to_ANGSTROM, DEGREES_per_RADIAN
        data = {}
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.ext_traj[:, i] *= DEGREES_per_RADIAN
            elif self.cv_type[i] == "distance":
                self.ext_traj[:, i] *= BOHR_to_ANGSTROM
            data[f"lambda{i}"] = self.ext_traj[:, i]
        return data

    def write_traj(self, filename: str = "CV_traj.dat"):
        """save trajectory for post-processing"""

        data = self._write_ext_traj()
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data, filename=filename)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
