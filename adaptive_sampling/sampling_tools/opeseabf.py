import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, diff_periodic
from .eabf import eABF
from .opes import OPES
from ..processing_tools.thermodynamic_integration import *


class OPESeABF(eABF, OPES, EnhancedSampling):
    """Well-Tempered Metadynamics extended-system Adaptive Biasing Force method

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased using a combination of ABF and OPES.

    Args:
        md: Object of the MD Interface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
            [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        ext_sigma: thermal width of coupling between collective and extended variable
            if None, it will be estimated based on the standard deviation of the CV in an initial MD
        ext_mass: mass of extended variable in atomic units
        adaptive_coupling_stride: initial MD steps to estimate ext_sigma
        adaptive_coupling_scaling: scaling factor for standard deviation of initial MD to ext_sigma
        adaptive_coupling_min: minimum for ext_sigma from adaptive estimate
        friction: friction coefficient for Langevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        nfull: Number of force samples per bin where full bias is applied,
            if nsamples < nfull the bias force is scaled down by nsamples/nfull
        kernel_std: standard deviation of first kernel,
            if None, kernel_std will be estimated from initial MD with `adaptive_std_freq*update_freq` steps
        update_freq: interval of md steps in which new kernels should be created
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol], default: 125.52 kJ/mol (30.0 kcal/mol)
        bandwidth_rescaling: if True, `kernel_std` shrinks during simulation to refine KDE
        adaptive_std: if adaptive kernel standard deviation is enabled, kernel_std will be updated according to std deviation of CV in MD
        adaptive_std_freq: time for estimation of standard deviation is set to `update_freq * adaptive_std_freq` MD steps
        explore: enables the OPES explore mode,
        normalize: normalize OPES probability density over explored space
        approximate_norm: enables linear scaling approximation of norm factor
        merge_threshold: threshold distance for kde-merging in units of std, `np.inf` disables merging
        recursive_merge: enables recursive merging until seperation of all kernels is above threshold distance
        bias_factor: bias factor of target distribution, default is `beta * energy_barr`
        numerical_forces: read forces from grid instead of calculating sum of kernels in every step, only for 1D CVs
        equil_temp: equilibrium temperature of MD
        verbose: print verbose information
        kinetics: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of CVs to the range of interest with harmonic walls
        output_freq: frequency in steps for writing outputs
        periodicity: periodicity of CVs, [[lower_boundary0, upper_boundary0], ...]
    """

    def __init__(
        self,
        *args,
        enable_abf: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.abf_forces = np.zeros_like(self.bias)
        self.enable_abf = enable_abf
        if self.verbose and self.enable_abf:
            print(f" >>> INFO: ABF enabled for OPES-eABF (N_full={self.nfull})")
        elif self.verbose:
            print(f" >>> INFO: ABF disabled. Running eOPES!")

    def step_bias(
        self,
        stabilize: bool = False,
        stabilizer_threshold: float = None,
        output_file: str = "opeseabf.out",
        traj_file: str = "CV_traj.dat",
        restart_file: str = "restart_opeseabf",
        **kwargs,
    ) -> np.ndarray:
        """Apply OPES-eABF to MD simulation

        Args:
            write_output: if on-the-fly free energy estimate and restart files should be written
            write_traj: if CV and extended system trajectory file should be written
            stabilize: if stabilisation algorithm should be applied for discontinous CVs
            stabilizer_threshold: treshold for stabilisation of extended system
            output_file: name of the output file
            traj_file: name of the trajectory file
            restart_file: name of the restart file

        Returns:
            bias_force: OPES-eABF biasing force of current step that has to be added to molecular forces
        """

        self.md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        if self.wandb_freq is not None and self.md_state.step % self.wandb_freq == 0:
            import wandb
            wandb.log(
                {
                    'amd/Xi0': xi[0],
                },
                commit=False
            )

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

            # setup OPES adaptive kernel std
            if self.initial_sigma_estimate or self.adaptive_std:
                self.adaptive_counter = self.adaptive_coupling_counter
                self.welford_mean = self.adaptive_coupling_mean
                self.welford_m2 = self.adaptive_coupling_m2
                self.welford_var = self.adaptive_coupling_var
                if (
                    self.initial_sigma_estimate
                    and self.adaptive_std_stride < self.adaptive_coupling_stride
                ):
                    # otherwise OPES sigma_0 would not be set any more
                    self.sigma_0 = self.ext_sigma / self.adaptive_coupling_scaling

        if stabilize and len(self.traj) > 0:
            self.stabilizer(xi, threshold=stabilizer_threshold)

        self._propagate()

        opes_force = self.opes_bias(np.copy(self.ext_coords))
        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.hill_std)
        force_sample = [0 for _ in range(2 * self.ncoords)]

        if self._check_boundaries(self.ext_coords):

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):

                if self.enable_abf:

                    # linear ramp function
                    ramp = (
                        1.0
                        if self.ext_hist[bin_la[1], bin_la[0]] > self.nfull
                        else self.ext_hist[bin_la[1], bin_la[0]] / self.nfull
                    )

                    # apply bias force on extended variable
                    force_sample[i] = self.ext_k[i] * diff_periodic(
                        self.ext_coords[i], xi[i], self.periodicity[i]
                    )
                    (
                        self.abf_forces[i][bin_la[1], bin_la[0]],
                        self.m2_force[i][bin_la[1], bin_la[0]],
                        self.var_force[i][bin_la[1], bin_la[0]],
                    ) = welford_var(
                        self.ext_hist[bin_la[1], bin_la[0]],
                        self.abf_forces[i][bin_la[1], bin_la[0]],
                        self.m2_force[i][bin_la[1], bin_la[0]],
                        force_sample[i],
                    )
                    self.ext_forces -= (
                        ramp * self.abf_forces[i][bin_la[1], bin_la[0]] - opes_force[i]
                    )
                else:
                    self.ext_forces += opes_force[i]

        # xi-conditioned accumulators for CZAR
        if self._check_boundaries(xi):
            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                force_sample[self.ncoords + i] = self.ext_k[i] * diff_periodic(
                    self.ext_coords[i], self.grid[i][bink[i]], self.periodicity[i]
                )
                self.correction_czar[i][bink[1], bink[0]] += force_sample[
                    self.ncoords + i
                ]

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        # Save values for traj
        if traj_file:
            self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
            self.traj = np.append(self.traj, [xi], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)

        self._up_momenta()

        if self.md_state.step % self.out_freq == 0:
            # write output

            if traj_file and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)
            if output_file:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"czar force {i}"] = self.czar_force[i]
                output[f"opespot"] = self.bias_potential
                self.write_output(output, filename=output_file)
            if restart_file:
                self.write_restart(filename=restart_file)
        return bias_force

    def reinit(self):
        """Reinit WTM-eABF and start building new bias potential"""
        self.histogram = np.zeros_like(self.histogram)
        self.bias = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.m2_force)
        self.ext_hist = np.zeros_like(self.ext_hist)
        self.correction_czar = np.zeros_like(self.correction_czar)
        self.abf_forces = np.zeros_like(self.abf_forces)
        self.center = []
        self.metapot = np.zeros_like(self.metapot)
        self.reinit_ext_system(self.traj[-1])

    def write_restart(self, filename: str = "restart_wtmeabf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
            abf_force=self.abf_forces,
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
            sum_weights=self.sum_weights,
            sum_weights_square=self.sum_weights2,
            norm_factor=self.norm_factor,
            height=self.kernel_height,
            center=self.kernel_center,
            sigma=self.kernel_std,
            explore=self.explore,
            n=self.n,
        )

    def restart(self, filename: str = "restart_wtmeabf", restart_ext_sys: bool = False):
        """restart from restart file

        Args:
            filename: name of restart file
            restart_ext_sys: restart coordinates and momenta of extended system
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        # restart eABF
        self.histogram = data["hist"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces = data["abf_force"]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords = data["ext_coords"]
        # restart OPES
        self.sum_weights = float(data["sum_weights"])
        self.sum_weights2 = float(data["sum_weights_square"])
        self.norm_factor = float(data["norm_factor"])
        self.kernel_height = data["height"]
        self.kernel_center = data["center"]
        self.kernel_std = data["sigma"]
        self.explore = data["explore"]
        self.n = data["n"]
        if self.verbose and self.md_state.step % self.update_freq == 0:
            print(f" >>> Info: Adaptive sampling restarted from {filename}!")

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
