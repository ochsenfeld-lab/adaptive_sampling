import numpy as np
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from .utils import correct_periodicity


class WTM(EnhancedSampling):
    """Well-Tempered Metadynamics

    see: Barducci et. al., Phys. rev. lett. (2008); https://doi.org/10.1103/PhysRevLett.100.020603

    An repulsive biasing potential is built by a superposition of Gaussian hills along the reaction coordinate.

    Args:
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CVs (can be Angstrom, Degree, or None)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if np.inf, hills are not scaled down (normal metadynamics)
        bias_factor: bias factor for WTM, if not None, overwrites `well_tempered_temp`
        force_from_grid: forces are accumulated on grid for performance (recommended),
                         if False, forces are calculated from sum of Gaussians in every step
    """

    def __init__(
        self,
        *args,
        hill_height: float = -1,
        hill_std: np.array = -1,
        hill_drop_freq: int = 100,
        well_tempered_temp: float = np.inf,
        bias_factor: float = None,
        force_from_grid: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        from ..units import kB_in_atomic, atomic_to_kJmol
        if hill_height <= 0:
            raise ValueError(" >>> Error: Hill height for WTM has to be > 0!")

        hill_std = (
            np.array([hill_std])
            if not hasattr(hill_std, "__len__")
            else np.asarray(hill_std)
        )  # convert to array if float
        if (hill_std <= 0).any():
            raise ValueError(
                " >>> Error: Hill standard deviation for WTM has to be > 0!"
            )

        if int(hill_drop_freq) <= 0:
            raise ValueError(" >>> Error: Update interval for WTM has to be int > 0!")

        if bias_factor is None and well_tempered_temp is None:
            raise ValueError(
                " >>> Error: Either bias_factor or well_tempered_temp has to be set!"
            )

        # general MtD parameters
        self.beta = 1.0 / (kB_in_atomic * self.equil_temp)
        self.verbose = verbose
        self.force_from_grid = force_from_grid
        self.hill_drop_freq = hill_drop_freq

        # Well-Tempered parameters
        if bias_factor is not None:
            self.bias_factor = bias_factor
            self.well_tempered_temp = (
                self.bias_factor * self.equil_temp - self.equil_temp
            )
        else:
            self.well_tempered_temp = well_tempered_temp
            self.bias_factor = self.well_tempered_temp / self.equil_temp + 1.0
        self.well_tempered = False if self.well_tempered_temp == np.inf else True
        self.wtm_prefac = (
            (self.equil_temp + self.well_tempered_temp) / self.well_tempered_temp
            if self.well_tempered
            else 1.0
        )
        if not self.well_tempered and self.verbose:
            print(" >>> Info: Well-tempered scaling of MtD hill_height switched off")
        elif self.well_tempered_temp <= 0:
            raise ValueError(
                " >>> Error: Effective temperature for Well-Tempered MtD has to be > 0!"
            )

        # hill parameters
        self.hill_std = self.unit_conversion_cv(np.asarray(hill_std))[0]
        self.hill_height = hill_height / atomic_to_kJmol
        self.hills_center = []
        self.hills_height = []
        self.hills_std = []

        # store results
        self.metapot = np.copy(self.histogram)
        self.bias_pot = 0.0
        self.mtd_rct = 0.0
        self.bias_pot_traj = []
        self.rct_traj = []

        if self.verbose:
            print(" >>> INFO: MtD Parameters:")
            print("\t ---------------------------------------------")
            print(f"\t Hill std:\t{self.hill_std}")
            print(f"\t Hill height:\t{self.hill_height * atomic_to_kJmol} kJ/mol")
            print(
                f"\t Bias factor:\t{self.bias_factor}\t\t({self.well_tempered_temp} K)"
            )
            print(f"\t Read force:\t{self.force_from_grid}")
            print("\t ---------------------------------------------")

    def step_bias(
        self,
        traj_file: str = "CV_traj.dat",
        out_file: str = "mtd.out",
        restart_file: str = "restart_mtd",
        **kwargs,
    ) -> np.array:
        """Apply MtD bias to MD

        Returns:
            bias_force: Bias force that has to be added to system forces
        """
        from ..units import atomic_to_kJmol
        self.md_state = self.the_md.get_sampling_data()
        (cv, grad_cv) = self.get_cv(**kwargs)

        # get mtd bias force
        bias_force = self.harmonic_walls(cv, grad_cv)
        mtd_forces = self.mtd_bias(cv)
        for i in range(self.ncoords):
            bias_force += mtd_forces[i] * grad_cv[i]

        # correction for kinetics
        if self.kinetics:
            self._kinetics(grad_cv)

        # store biased histogram along CV for output
        if out_file and self._check_boundaries(cv):
            bink = self.get_index(cv)
            self.histogram[bink[1], bink[0]] += 1

        # shared-bias metadynamics
        if self.shared:
            self.shared_bias(**kwargs)

        # Save values for traj output
        if traj_file:
            self.traj = np.append(self.traj, [cv], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)
            self.bias_pot_traj.append(self.bias_pot)
            self.rct_traj.append(self.mtd_rct)

        # Write output
        if self.md_state.step % self.out_freq == 0:
            if traj_file and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)
            if out_file:
                self.pmf = self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf * atomic_to_kJmol,
                    "MtD Pot": self.metapot * atomic_to_kJmol,
                }
                self.write_output(output, filename=out_file)
            if restart_file:
                self.write_restart(filename=restart_file)

        return bias_force

    def get_pmf(self) -> np.array:
        """Calculate current PMF estimate on `self.grid`

        Returns:
            pmf: current PMF estimate from Mtd kernels
        """
        pmf = -self.metapot
        if self.well_tempered:
            pmf *= self.wtm_prefac
        pmf -= pmf.min()
        return pmf

    def write_traj(self, filename="CV_traj.dat"):
        data = {
            "Epot [Ha]": self.epot,
            "T [K]": self.temp,
            "Biaspot [Ha]": self.bias_pot_traj,
            "C(t) [Ha]": self.rct_traj,
        }
        self._write_traj(data, filename=filename)

        # Reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.bias_pot_traj = [self.bias_pot_traj[-1]]
        self.rct_traj = [self.rct_traj[-1]]

    def shared_bias(
        self,
        sync_interval: int = 50,
        mw_file: str = "shared_bias",
        local_file: str = "restart_wtmeabf_local",
        n_trials: int = 10,
        sleep_time: int = 0.1,
    ):
        """Multiple walker shared-bias implementation for metadynamics

        Args:
            sync_interval: number of steps between bias syncs
            mw_file: filename for buffer .npz file
            n_trials: maximum number of sync trials to open the buffer file before aborting
            sleep_time: sleep time before new trial to open buffer file
        """
        import os, time

        md_state = self.the_md.get_sampling_data()
        if md_state.step == 0:

            if sync_interval % self.hill_drop_freq != 0:
                raise ValueError(
                    " >>> ERROR: Sync interval for shared-bias WTM has to divisible through the frequency of hill creation!"
                )

            if self.verbose:
                print(
                    " >>> Info: Creating a new instance for shared-bias metadynamics."
                )
                print(
                    " >>> Info: Data of local walker stored in `restart_wtm_local.npz`."
                )

            # create seperate restart file with local data only
            self.write_restart(
                filename=local_file,
            )

            self.num_hills_last_sync = len(self.hills_center)
            self.hist_last_sync = np.copy(self.histogram)
            if not os.path.isfile(mw_file + ".npz"):
                if self.verbose:
                    print(
                        f" >>> Info: Creating buffer file for shared-bias metadynamics: `{mw_file}.npz`."
                    )
                self.write_restart(filename=mw_file)
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(
                    f" >>> Info: Syncing with existing buffer file for shared-bias metadynamics: `{mw_file}.npz`."
                )

        if md_state.step % sync_interval == 0:

            # get new hills
            new_center = np.asarray(self.hills_center)[self.num_hills_last_sync :]
            new_height = np.asarray(self.hills_height)[self.num_hills_last_sync :]
            new_std = np.asarray(self.hills_std)[self.num_hills_last_sync :]

            delta_hist = self.histogram - self.hist_last_sync

            self._update_wtm(
                local_file,
                delta_hist,
                new_center,
                new_height,
                new_std,
            )

            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):

                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666)
                    self._update_wtm(
                        mw_file,
                        delta_hist,
                        new_center,
                        new_height,
                        new_std,
                    )
                    self.restart(filename=mw_file)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again
                    self.num_hills_last_sync = len(self.hills_center)
                    self.hist_last_sync = np.copy(self.histogram)
                    if self.verbose:
                        print(
                            f" >>> Info: Synced bias with `{mw_file}.npz` after {trial} attempts."
                        )
                    break

                elif trial < n_trials:
                    if self.verbose:
                        print(
                            f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts."
                        )
                    time.sleep(sleep_time)
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

    def _update_wtm(
        self,
        filename: str,
        delta_hist: np.array,
        new_hill_centers: np.array,
        new_hill_heights: np.array,
        new_hill_stds: np.array,
    ):
        """updates shared bias buffer"""
        with np.load(f"{filename}.npz") as data:
            new_hist = data["hist"] + delta_hist
            new_heights = np.append(data["height"], new_hill_heights)
            if self.ncoords == 1:
                new_centers = np.append(data["center"], new_hill_centers)
                new_stds = np.append(data["sigma"], new_hill_stds)
            else:
                new_centers = np.vstack((data["center"], new_hill_centers))
                new_stds = np.vstack((data["sigma"], new_hill_stds))

        self._write_restart(
            filename=filename,
            hist=new_hist,
            height=new_heights,
            center=new_centers,
            sigma=new_stds,
        )

    def write_restart(self, filename: str = "restart_mtd"):
        """Dumps state of MtD to restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            height=self.hills_height,
            center=self.hills_center,
            sigma=self.hills_std,
        )

    def restart(self, filename: str = "restart_mtd"):
        """Restart MtD from previous simulation

        Args:
            filename: name of restart
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> ERROR: restart file {filename}.npz not found!")

        # Load dictionary entries from restart file
        self.histogram = data["hist"]
        self.hills_height = data["height"].tolist()
        self.hills_center = data["center"].tolist()
        self.hills_std = data["sigma"].tolist()
        if not hasattr(self.hills_center[0], "__len__"):
            self.hills_center = [np.array([c]) for c in self.hills_center]
            self.hills_std = [np.array([std]) for std in self.hills_std]
        if (
            self.verbose
            and self.md_state.step % self.hill_drop_freq == 0
            and not self.shared_bias
        ):
            print(f" >>> Info: Adaptive sampling restarted from {filename}!")

    def mtd_bias(self, cv: np.array) -> np.array:
        """Calculate MtD bias force from KDE of hills
        which is updated according to `self.update_freq`

        Args:
            cv: new value of CV

        Returns:
            bias force: len(ncoords) array of bias forces
        """
        # get bias potential and forces
        if self.force_from_grid and self._check_boundaries(cv):
            idx = self.get_index(cv)
            self.bias_pot = self.metapot[idx[1], idx[0]]
            mtd_force = [self.bias[i][idx[1], idx[0]] for i in range(self.ncoords)]
        else:
            gaussians, derivative = self.calc_hills(cv, requires_grad=True)
            self.bias_pot = np.sum(gaussians)
            mtd_force = derivative

        # MtD update
        if self.md_state.step % self.hill_drop_freq == 0:
            self.update_kde(cv)

        return np.asarray(mtd_force)

    def calc_hills(self, cv, requires_grad: bool = False) -> np.array:
        """Get values of gaussian hills and optionally the gradient of the associated potential

        Args:
            cv: value of CV where the kernels should be evaluated
            requires_grad: if True, gradient of mtd potential is returned as second argument

        Returns:
            hills: values of hills at CV
            derivative: derivative of WTM potential, only returned if requires_grad
        """

        if len(self.hills_center) == 0:
            if requires_grad:
                return 0.0, np.zeros(self.ncoords)
            return 0.0

        # distance to kernel centers
        s_diff = cv - np.asarray(self.hills_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        # evaluate values of kernels at cv
        hills = np.asarray(self.hills_height) * np.exp(
            -0.5
            * np.sum(np.square(np.divide(s_diff, np.asarray(self.hills_std))), axis=1)
        )
        if requires_grad:
            derivative = np.sum(
                -hills * np.divide(s_diff, np.square(np.asarray(self.hills_std))).T,
                axis=1,
            )
            return hills, derivative

        return hills

    def update_kde(self, cv: np.array):
        """on-the-fly update of kernel density estimation of pmf along CVs

        Args:
            CV: new value of CVS
        """
        self._add_kernel(self.hill_height, cv, self.hill_std)
        self._update_grid_potential(
            self.hills_height[-1], self.hills_center[-1], self.hills_std[-1]
        )
        self._get_rct()

    def _add_kernel(self, h_new: float, s_new: np.array, std_new: np.array):
        """Add new Kernel to KDE

        Args:
            h_new: hills height
            s_new: hills position
            std_new: hills standard deviation
        """
        from ..units import kB_in_atomic
        if self.well_tempered:
            w = h_new * np.exp(
                -self.bias_pot / (kB_in_atomic * self.well_tempered_temp)
            )
        else:
            w = h_new

        self.hills_height.append(w)
        self.hills_center.append(s_new)
        self.hills_std.append(std_new)

    def _get_rct(self):
        """get reweighting factor for metadynamics"""
        minusBetaF = (
            self.beta * self.bias_factor / (self.bias_factor - 1.0)
            if self.well_tempered
            else self.beta
        )
        minusBetaFplusV = (
            self.beta / (self.bias_factor - 1.0) if self.well_tempered else 0.0
        )
        max_bias = self.metapot.max()  # to avoid overflow
        Z_0 = np.sum(np.exp(minusBetaF * (self.metapot - max_bias)))
        Z_V = np.sum(np.exp(minusBetaFplusV * (self.metapot - max_bias)))
        self.mtd_rct = (1.0 / self.beta) * np.log(Z_0 / Z_V) + max_bias

    def _update_grid_potential(
        self, height_new: np.array, center_new: np.array, std_new: np.array
    ):
        """Add new kernel to `self.metapot` and `self.bias`

        Args:
            height_new: height of new kernel
            center_new: center of new kernel
            std_new: standard deviation of new kernel
        """
        if self.ncoords == 1:
            grid_full = np.asarray(self.grid[0]).reshape((-1, 1))
        elif self.ncoords == 2:
            xx, yy = np.meshgrid(self.grid[0], self.grid[1])
            grid_full = np.stack([xx, yy], axis=-1)
        else:
            raise NotImplementedError(
                " >>> ERROR: MtD grid potential not implemented for ncoords > 2"
            )

        s_diff = np.rollaxis(grid_full - np.asarray(center_new), -1)
        for i in range(self.ncoords):
            s_diff[i, :] = correct_periodicity(s_diff[i, :], self.periodicity[i])
        pot_new = (
            np.asarray(height_new)
            * np.exp(
                -0.5
                * np.sum(np.square(np.divide(s_diff.T, np.asarray(std_new))), axis=-1)
            ).T
        )
        self.metapot += pot_new
        for i in range(self.ncoords):
            self.bias[i] -= pot_new * np.divide(s_diff[i], np.square(std_new[i]))
