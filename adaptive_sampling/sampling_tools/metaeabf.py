import os, time
import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, combine_welford_stats, diff_periodic
from .eabf import eABF
from .metadynamics import WTM


class WTMeABF(eABF, WTM, EnhancedSampling):
    """Well-Tempered Metadynamics extended-system Adaptive Biasing Force method

    References:
       Fu et. al., J. Phys. Chem. Lett. (2018); https://doi.org/10.1021/acs.jpclett.8b01994

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particle is biased using a combination of ABF and Metadynamics.

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
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CVs (can be Angstrom, Degree, or None)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if np.inf, hills are not scaled down (normal metadynamics)
        bias_factor: bias factor for WTM, if not None, overwrites `well_tempered_temp`
        force_from_grid: forces are accumulated on grid for performance (recommended),
                         if False, forces are calculated from sum of Gaussians in every step
        equil_temp: equilibrium temperature of MD
        verbose: print verbose information
        kinetic: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """

    def __init__(self, *args, enable_abf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_abf = enable_abf
        self.abf_forces = np.zeros_like(self.bias)
        if self.verbose and self.apply_abf:
            print(f" >>> INFO: ABF enabled for WTM-eABF (N_full={self.nfull})")
        elif self.verbose:
            print(f" >>> INFO: ABF disabled. Running eWTM!")

    def step_bias(
        self,
        write_output: bool = True,
        write_traj: bool = True,
        stabilize: bool = False,
        stabilizer_threshold: float = None,
        output_file: str = "wtmeabf.out",
        traj_file: str = "CV_traj.dat",
        restart_file: str = "restart_wtmeabf",
        **kwargs,
    ) -> np.ndarray:
        """Apply WTM-eABF to MD simulation

        Args:
            write_output: if on-the-fly free energy estimate and restart files should be written
            write_traj: if CV and extended system trajectory file should be written
            stabilize: if stabilisation algorithm should be applied for discontinuous CVs
            stabilizer_threshold: treshold for stabilisation of extended system
            output_file: name of the output file
            traj_file: name of the trajectory file
            restart_file: name of the restart file

        Returns:
            bias_force: WTM-eABF biasing force of current step that has to be added to molecular forces
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

        mtd_forces = self.mtd_bias(self.ext_coords)
        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.hill_std)
        force_sample = [0 for _ in range(2 * self.ncoords)]

        if self._check_boundaries(self.ext_coords):

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):

                if self.apply_abf:
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
                        ramp * self.abf_forces[i][bin_la[1], bin_la[0]] - mtd_forces[i]
                    )
                else:
                    self.ext_forces += mtd_forces[i]

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

        # shared-bias eABF
        if self.shared:
            self.shared_bias(
                xi,
                force_sample,
                **kwargs,
            )

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

            if write_traj and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"metaforce {i}"] = self.bias[i]
                    output[f"abf force {i}"] = self.abf_forces[i]
                    output[f"czar force {i}"] = self.czar_force[i]
                output[f"metapot"] = self.metapot

                self.write_output(output, filename=output_file)
                self.write_restart(filename=restart_file)

        return bias_force

    def shared_bias(
        self,
        xi,
        force_sample,
        sync_interval: int = 1000,
        mw_file: str = "../shared_bias",
        local_file: str = "restart_wtmeabf_local",
        n_trials: int = 100,
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
        # create buffer file and accumulators for new samples
        if self.md_state.step == 0:
            if self.verbose:
                print(" >>> Info: Creating a new instance for shared-bias eABF.")
                print(f" >>> Info: Data of local walker stored in `{local_file}.npz`.")

            # create seperate restart file with local data only
            self._write_restart(
                filename=local_file,
                hist=self.histogram,
                force=self.bias,
                m2=self.m2_force,
                ext_hist=self.ext_hist,
                czar_corr=self.correction_czar,
                abf_force=self.abf_forces,
                height=self.hills_height,
                center=self.hills_center,
                sigma=self.hills_std,
            )

            if sync_interval % self.hill_drop_freq != 0:
                raise ValueError(
                    " >>> Fatal Error: Sync interval for shared-bias WTM has to divisible through the frequency of hill creation!"
                )

            self.num_hills_last_sync = len(self.hills_center)
            self.hist_new_samples = np.zeros_like(self.histogram)
            self.m2_new_samples = np.zeros_like(self.m2_force)
            self.abf_forces_new_samples = np.zeros_like(self.abf_forces)
            self.ext_hist_new_samples = np.zeros_like(self.ext_hist)
            self.czar_corr_new_samples = np.zeros_like(self.correction_czar)

            if not os.path.isfile(mw_file + ".npz"):
                if self.verbose:
                    print(
                        f" >>> Info: Creating buffer file for shared-bias WTM-eABF: `{mw_file}.npz`."
                    )
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    force=self.bias,
                    m2=self.m2_force,
                    ext_hist=self.ext_hist,
                    czar_corr=self.correction_czar,
                    abf_force=self.abf_forces,
                    height=self.hills_height,
                    center=self.hills_center,
                    sigma=self.hills_std,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(f" >>> Info: Syncing with buffer file `{mw_file}.npz`.")

        # add new sample to accumulators of new data since last sync
        if self._check_boundaries(self.ext_coords):
            bin_la = self.get_index(self.ext_coords)
            self.ext_hist_new_samples[bin_la[1], bin_la[0]] += 1
            for j in range(self.ncoords):
                (
                    self.abf_forces_new_samples[j][bin_la[1], bin_la[0]],
                    self.m2_new_samples[j][bin_la[1], bin_la[0]],
                    _,
                ) = welford_var(
                    self.ext_hist_new_samples[bin_la[1], bin_la[0]],
                    self.abf_forces_new_samples[j][bin_la[1], bin_la[0]],
                    self.m2_new_samples[j][bin_la[1], bin_la[0]],
                    force_sample[j],
                )

            if self._check_boundaries(xi):
                bin_xi = self.get_index(xi)
                self.hist_new_samples[bin_xi[1], bin_xi[0]] += 1
                for j in range(self.ncoords):
                    self.czar_corr_new_samples[j][bin_xi[1], bin_xi[0]] += force_sample[
                        self.ncoords + j
                    ]

        # update buffer file
        if self.md_state.step % sync_interval == 0:

            # get new hills
            new_center = np.asarray(self.hills_center)[self.num_hills_last_sync :]
            new_height = np.asarray(self.hills_height)[self.num_hills_last_sync :]
            new_std = np.asarray(self.hills_std)[self.num_hills_last_sync :]

            # add new samples to local restart
            self._update_wtmeabf(
                local_file,
                self.hist_new_samples,
                self.ext_hist_new_samples,
                self.abf_forces_new_samples,
                self.m2_new_samples,
                self.czar_corr_new_samples,
                new_height,
                new_center,
                new_std,
            )

            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):

                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666)
                    self._update_wtmeabf(
                        mw_file,
                        self.hist_new_samples,
                        self.ext_hist_new_samples,
                        self.abf_forces_new_samples,
                        self.m2_new_samples,
                        self.czar_corr_new_samples,
                        new_height,
                        new_center,
                        new_std,
                    )
                    self.restart(filename=mw_file, restart_ext_sys=False)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again

                    self.get_pmf()  # get new global pmf

                    # reset new sample accumulators
                    self.num_hills_last_sync = len(self.hills_height)
                    self.hist_new_samples = np.zeros_like(self.histogram)
                    self.m2_new_samples = np.zeros_like(self.m2_force)
                    self.abf_forces_new_samples = np.zeros_like(self.abf_forces)
                    self.ext_hist_new_samples = np.zeros_like(self.ext_hist)
                    self.czar_corr_new_samples = np.zeros_like(self.correction_czar)
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

            # recalculate `self.metapot` and `self.bias` to ensure convergence of WTM potential
            if self.ncoords == 1:
                grid_full = np.asarray(self.grid[0]).reshape((-1, 1))
            elif self.ncoords == 2:
                xx, yy = np.meshgrid(self.grid[0], self.grid[1])
                grid_full = np.stack([xx.flatten(), yy.flatten()], axis=1)

            shape = self.metapot.shape
            self.metapot = np.zeros_like(self.metapot).flatten()
            self.bias = np.zeros_like(self.bias).reshape(
                (len(self.metapot), self.ncoords)
            )
            for i, bin_coords in enumerate(grid_full):
                pot, der = self.calc_hills(bin_coords, requires_grad=True)
                self.metapot[i] = np.sum(pot)
                self.bias[i] = der
            self.metapot = self.metapot.reshape(shape)
            self.bias = np.rollaxis(self.bias.reshape(shape + (self.ncoords,)), -1)

    def _update_wtmeabf(
        self,
        filename: str,
        hist: np.ndarray,
        ext_hist: np.ndarray,
        abf_forces: np.ndarray,
        m2: np.ndarray,
        czar_corr: np.ndarray,
        new_hill_heights: np.ndarray,
        new_hill_centers: np.ndarray,
        new_hill_stds: np.ndarray,
    ):
        with np.load(f"{filename}.npz") as data:

            new_hist = data["hist"] + hist
            new_czar_corr = data["czar_corr"] + czar_corr

            new_m2 = np.zeros_like(self.m2_force).flatten()
            new_abf_forces = np.zeros_like(self.abf_forces).flatten()
            new_ext_hist = np.zeros_like(self.ext_hist).flatten()
            new_heights = np.append(data["height"], new_hill_heights)
            if self.ncoords == 1:
                new_centers = np.append(data["center"], new_hill_centers)
                new_stds = np.append(data["sigma"], new_hill_stds)
            else:
                new_centers = np.vstack((data["center"], new_hill_centers))
                new_stds = np.vstack((data["sigma"], new_hill_stds))

            for i in range(len(hist.flatten())):
                (
                    new_ext_hist[i],
                    new_abf_forces[i],
                    new_m2[i],
                    _,
                ) = combine_welford_stats(
                    data["ext_hist"].flatten()[i],
                    data["abf_force"].flatten()[i],
                    data["m2"].flatten()[i],
                    ext_hist.flatten()[i],
                    abf_forces.flatten()[i],
                    m2.flatten()[i],
                )

        self._write_restart(
            filename=filename,
            hist=new_hist,
            m2=new_m2.reshape(self.m2_force.shape),
            ext_hist=new_ext_hist.reshape(self.histogram.shape),
            czar_corr=new_czar_corr,
            abf_force=new_abf_forces.reshape(self.abf_forces.shape),
            height=new_heights,
            center=new_centers,
            sigma=new_stds,
        )

    def reinit(self):
        """Reinit WTM-eABF and start building new bias potential"""
        self.histogram = np.zeros_like(self.histogram)
        self.bias = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.m2_force)
        self.ext_hist = np.zeros_like(self.ext_hist)
        self.correction_czar = np.zeros_like(self.correction_czar)
        self.abf_forces = np.zeros_like(self.abf_forces)
        self.hills_center = []
        self.hills_height = []
        self.hills_std = [] = []
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
            height=self.hills_height,
            center=self.hills_center,
            sigma=self.hills_std,
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
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

        self.histogram = data["hist"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces = data["abf_force"]
        self.hills_height = data["height"].tolist()
        self.hills_center = data["center"].tolist()
        self.hills_std = data["sigma"].tolist()
        if not hasattr(self.hills_center[0], "__len__"):
            self.hills_center = [np.array([c]) for c in self.hills_center]
            self.hills_std = [np.array([std]) for std in self.hills_std]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords = data["ext_coords"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

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
