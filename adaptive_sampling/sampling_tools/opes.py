import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import correct_periodicity
from .utils import welford_var

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling

    References:
        OPES: M. Invernizzi, M. Parrinello; J. Phys. Chem. 2020; <https://doi.org/10.1021/acs.jpclett.0c00497>
        OPES Explore: M. Invernizzi, M. Parrinello; J. Chem. Theory. Comput. 2022; <https://doi.org/10.1021/acs.jctc.2c00152>

    Args:
        md: Object of the MD Interface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
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
        recursive_merge: enables recursive merging until separation of all kernels is above threshold distance
        bias_factor: bias factor of target distribution, default is `beta * energy_barr`
        force_from_grid: read forces from grid instead of calculating sum of kernels in every step
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
        kernel_std: np.array = None,
        update_freq: int = 500,
        energy_barr: float = 30.0 / 0.239006,  # 30 kcal/mol
        bandwidth_rescaling: bool = True,
        adaptive_std: bool = False,
        adaptive_std_freq: int = 10,
        explore: bool = False,
        normalize: bool = True,
        approximate_norm: bool = True,
        merge_threshold: float = 1.0,
        recursive_merge: bool = True,
        bias_factor: float = None,
        force_from_grid: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from ..units import kB_in_atomic, atomic_to_kJmol

        # kernel standard deviation
        self.bandwidth_rescaling = bandwidth_rescaling
        self.adaptive_std = adaptive_std
        self.adaptive_std_stride = adaptive_std_freq * update_freq
        self.adaptive_counter = 0
        self.welford_m2 = np.zeros(self.ncoords)
        self.welford_mean = np.zeros(self.ncoords)

        if not hasattr(kernel_std, "__len__") and not kernel_std:
            self.initial_sigma_estimate = True
        elif hasattr(kernel_std, "__len__") and any(std is None for std in kernel_std):
            self.initial_sigma_estimate = True
        else:
            self.initial_sigma_estimate = False

        if self.initial_sigma_estimate:
            self.sigma_0 = np.full((self.ncoords,), np.nan)
        elif not hasattr(kernel_std, "__len__"):
            self.sigma_0 = self.unit_conversion_cv(np.asarray([kernel_std]))[0]
        else:
            self.sigma_0 = self.unit_conversion_cv(np.asarray(kernel_std))[0]

        # other parameters
        self.explore = explore
        self.update_freq = update_freq
        self.approximate_norm = approximate_norm
        self.merge_threshold = merge_threshold
        self.merge = False if self.merge_threshold == np.inf else True
        if self.verbose and self.merge_threshold <= 0.0:
            raise ValueError(" >>> OPES: Merge threshold should be > 0")
        self.recursive_merge = recursive_merge
        self.bias_factor = bias_factor
        self.normalize = normalize
        self.numerical_forces = force_from_grid

        self.beta = 1.0 / (kB_in_atomic * self.equil_temp)
        self.energy_barr = energy_barr / atomic_to_kJmol
        if not energy_barr > 0.0:
            raise ValueError(" >>> OPES: The barrier should be > 0 ")
        self.gamma = (
            self.beta * self.energy_barr if bias_factor == None else bias_factor
        )
        if self.gamma <= 1.0:
            raise ValueError(" >>> OPES: The bias factor should be > 1")
        if self.gamma == np.inf:
            self.gamma_prefac = 1.0
            self.gamma = np.inf
        else:
            self.gamma_prefac = self.gamma - 1 if self.explore else 1 - 1 / self.gamma

        self.epsilon = np.exp((-self.beta * self.energy_barr) / self.gamma_prefac)
        if self.epsilon < 0.0:
            raise ValueError(
                " >>> OPES: epsilon needs to be positive. The `energy_barr` might be too high."
            )
        if self.normalize:
            self.norm_factor = np.power(self.epsilon, self.gamma_prefac)
        else:
            self.norm_factor = 1.0

        self.n = 0
        self.sum_weights = (
            np.power(self.epsilon, self.gamma_prefac) if self.normalize else 1.0
        )
        self.sum_weights2 = np.square(self.sum_weights) if self.normalize else 1.0
        self.n_eff = (
            np.square(1.0 + self.sum_weights) / (1.0 + self.sum_weights2)
            if not self.explore
            else self.n
        )
        self.KDE_norm = self.sum_weights if not self.explore else self.n
        self.old_KDE_norm = np.copy(self.KDE_norm)
        self.old_nker = 0
        self.rct = (1.0 / self.beta) * np.log(self.sum_weights)

        self.kernel_center = []
        self.kernel_height = []
        self.kernel_std = []
        self.bias_pot_traj = []
        self.rct_traj = []
        self.zed_traj = []

        self.bias_potential = np.copy(self.histogram)
        self.bias_pot = 0.0
        self.bias_pot_traj = []

        if self.verbose:
            print(" >>> INFO: OPES Parameters:")
            print("\t ---------------------------------------------")
            print(
                f"\t Kernel_std:\t{self.sigma_0 if not self.initial_sigma_estimate else f'estimate from {self.adaptive_std_stride} steps'}"
            )
            print(f"\t Rescaling:\t{self.bandwidth_rescaling}")
            print(
                f"\t Adaptive:\t{self.adaptive_std}\t({self.adaptive_std_stride} steps)"
            )
            print(
                f"\t Normalize:\t{self.normalize}\t(approximated: {self.approximate_norm})"
            )
            print(f"\t Explore:\t{self.explore}")
            print(
                f"\t Barrier:\t{self.energy_barr*atomic_to_kJmol*kJ_to_kcal} kcal/mol"
            )
            print(f"\t Bias factor:\t{self.gamma}")
            print(f"\t Read force:\t{self.numerical_forces}")
            print(
                f"\t Kernel merge:\t{self.merge}\t(threshold: {self.merge_threshold})"
            )
            print("\t ---------------------------------------------")

    def step_bias(
        self,
        traj_file: str = "CV_traj.dat",
        out_file: str = "opes.out",
        restart_file: str = "restart_opes",
        **kwargs,
    ) -> np.array:
        """Apply OPES bias to MD

        Returns:
            bias_force: Bias force that has to be added to system forces
        """
        from ..units import atomic_to_kJmol
        self.md_state = self.the_md.get_sampling_data()
        (cv, grad_cv) = self.get_cv(**kwargs)

        if self.kinetics:
            self._kinetics(grad_cv)

        # get OPES bias force
        forces = self.opes_bias(cv)
        bias_force = self.harmonic_walls(cv, grad_cv)
        for i in range(self.ncoords):
            bias_force += forces[i] * grad_cv[i]

        # store biased histogram along CV for output
        if out_file and self._check_boundaries(cv):
            bink = self.get_index(cv)
            self.histogram[bink[1], bink[0]] += 1

        # Save values for traj output
        if traj_file:
            self.traj = np.append(self.traj, [cv], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)
            self.bias_pot_traj.append(self.bias_pot)
            self.zed_traj.append(self.norm_factor)
            self.rct_traj.append(self.rct)

        # Write output
        if self.md_state.step % self.out_freq == 0:
            if traj_file and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)
            if out_file:
                self.pmf = self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf * atomic_to_kJmol,
                    "OPES Pot": self.bias_potential * atomic_to_kJmol,
                }
                self.write_output(output, filename=out_file)
            if restart_file:
                self.write_restart(filename=restart_file)

        return bias_force

    def get_pmf(self) -> np.array:
        """Calculate current PMF estimate on `self.grid`

        Returns:
            pmf: current PMF estimate from OPES kernels
        """
        pmf = (
            -self.bias_potential / self.gamma_prefac
            if not self.explore
            else -self.bias_potential
        )
        pmf -= pmf.min()
        return pmf

    def write_traj(self, filename="CV_traj.dat"):
        data = {
            "Epot [Ha]": self.epot,
            "T [K]": self.temp,
            "Biaspot [Ha]": self.bias_pot_traj,
            "Zed": self.zed_traj,
            "C(t) [Ha]": self.rct_traj,
        }
        self._write_traj(data, filename=filename)

        # Reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.bias_pot_traj = [self.bias_pot_traj[-1]]
        self.zed_traj = [self.zed_traj[-1]]
        self.rct_traj = [self.rct_traj[-1]]

    def shared_bias(self):
        raise ValueError(
            " >>> ERROR: Multiple-walker shared bias not available for OPES!"
        )

    def write_restart(self, filename: str = "restart_opes"):
        """Dumps state of OPES to restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            sum_weights=self.sum_weights,
            sum_weights_square=self.sum_weights2,
            norm_factor=self.norm_factor,
            height=self.kernel_height,
            center=self.kernel_center,
            sigma=self.kernel_std,
            explore=self.explore,
            n=self.n,
        )

    def restart(self, filename: str = "restart_opes"):
        """Restart OPES from previous simulation

        Args:
            filename: name of restart
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> ERROR: restart file {filename}.npz not found!")

        # Load dictionary entries from restart file
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

    def opes_bias(self, cv: np.array) -> np.array:
        """Calculate OPES bias force from KDE of probability density
        which is updated according to `self.update_freq`

        Args:
            cv: new value of CV

        Returns:
            bias force: len(ncoords) array of bias forces
        """
        # adaptive kernel `sigma_0` estimate from MD
        if (
            self.initial_sigma_estimate
            and self.md_state.step < self.adaptive_std_stride
        ):
            self.sigma_0 = self.estimate_kernel_std(cv)
            return np.zeros_like(self.the_md.coords)

        elif self.initial_sigma_estimate or self.adaptive_std:
            std = self.estimate_kernel_std(cv)
            if self.initial_sigma_estimate:
                self.sigma_0 = std
                self.initial_sigma_estimate = False
                if self.verbose:
                    print(
                        f" >>> INFO: kernel standard deviation for OPES is set to {self.sigma_0}"
                    )
            elif self.md_state.step >= self.adaptive_std_stride:
                self.sigma_0 = std
                if not self.explore:
                    self.sigma_0 /= np.sqrt(self.gamma)

        # get bias potential and forces
        if self.numerical_forces and self._check_boundaries(cv):
            idx = self.get_index(cv)
            self.bias_pot = self.bias_potential[idx[1], idx[0]]
            opes_force = [self.bias[i][idx[1], idx[0]] for i in range(self.ncoords)]
        else:
            gaussians, kde_derivative = self.calc_gaussians(cv, requires_grad=True)
            self.prob_dist = np.sum(gaussians) / self.KDE_norm
            self.bias_pot = self.calc_potential(self.prob_dist)
            opes_force = self.calc_forces(self.prob_dist, kde_derivative)

        # OPES KDE update
        if self.md_state.step % self.update_freq == 0:
            self.update_kde(cv)

        return opes_force

    def update_kde(self, cv: np.array):
        """on-the-fly update of kernel density estimation of probability density along CVs

        Args:
            CV: new value of CVS
        """
        self.delta_kernel_height = []
        self.delta_kernel_center = []
        self.delta_kernel_sigma = []

        self.n += 1  # counter for total kernels

        # Calculate weight coefficients
        weight_coeff = np.exp(self.beta * self.bias_pot)
        self.sum_weights += weight_coeff
        self.sum_weights2 += weight_coeff * weight_coeff
        self.KDE_norm = self.sum_weights if not self.explore else self.n

        # Bandwidth rescaling
        self.n_eff = (
            np.square(1.0 + self.sum_weights) / (1.0 + self.sum_weights2)
            if not self.explore
            else self.n
        )
        self.rct = (1.0 / self.beta) * np.log(self.sum_weights / self.n)

        if self.bandwidth_rescaling and len(self.kernel_std) > 0:
            sigma_i = self.sigma_0 * np.power(
                (self.n_eff * (self.ncoords + 2.0) / 4.0), -1.0 / (self.ncoords + 4.0)
            )
        else:
            sigma_i = np.copy(self.sigma_0)

        height = np.prod(self.sigma_0 / sigma_i)
        if not self.explore:
            height *= weight_coeff

        # Kernel Density
        self.add_kernel(height, cv, sigma_i)

        # Calculate normalization factor
        if self.normalize:
            self.norm_factor = self.calc_norm_factor(approximate=self.approximate_norm)
        self.grid_potential()

    def calc_gaussians(self, cv, requires_grad: bool = False) -> np.array:
        """Get normalized value of gaussian hills

        Args:
            cv: value of CV where the kernels should be evaluated
            requires_grad: if True, accumulated gradient of KDE is returned as second argument

        Returns:
            gaussians: values of gaussians at CV
            kde_derivative: derivative of KDE, only if requires_grad
        """

        if len(self.kernel_center) == 0:
            if requires_grad:
                return 0.0, np.zeros(self.ncoords)
            return 0.0

        # distance to kernel centers
        s_diff = cv - np.asarray(self.kernel_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        # evaluate values of kernels at cv
        gaussians = np.asarray(self.kernel_height) * np.exp(
            -0.5
            * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_std))), axis=1)
        )
        if requires_grad:
            kde_derivative = (
                np.sum(
                    -gaussians
                    * np.divide(s_diff, np.square(np.asarray(self.kernel_std))).T,
                    axis=1,
                )
                / self.KDE_norm
            )
            return gaussians, kde_derivative

        return gaussians

    def add_kernel(self, h_new: float, s_new: np.array, std_new: np.array):
        """Add new Kernel to KDE

        Args:
            h_new: kernel height
            s_new: kernel position
            std_new: kernel standard deviation
        """
        self.kernel_height.append(h_new)
        self.kernel_center.append(s_new)
        self.kernel_std.append(std_new)

        kernel_min_ind, min_distance = self.calc_min_dist(s_new)

        # Recursive merging if enabled and distances under threshold
        while (
            self.merge
            and np.all(min_distance < self.merge_threshold)
            and len(self.kernel_center) > 1
        ):

            # Merge again
            h_new, s_new, std_new = self.merge_kernels(
                kernel_min_ind, h_new, s_new, std_new
            )

            # Calculate new distances to update while condition
            kernel_min_ind, min_distance = self.calc_min_dist(s_new)

            if not self.recursive_merge:
                break

        # Append final kernel delta list
        self.delta_kernel_height.append(h_new)
        self.delta_kernel_center.append(s_new)
        self.delta_kernel_sigma.append(std_new)

    def merge_kernels(
        self, kernel_min_ind: int, h_new: float, s_new: np.array, std_new: np.array
    ) -> tuple:
        """Merge two kernels

        Args:
            kernel_min_ind: index of kernel that should be merged with new kernel
            h_new: new kernel height
            s_new: new kernel position
            std_new: new kernel standard deviation

        Returns:
            h_merged: height of merged kernel
            s_merged: position of merged kernel
            std_merged: standard deviation of merged kernel
        """
        # Calculate properties of merged kernel
        h_merge = self.kernel_height[kernel_min_ind] + h_new
        s_merge = (1.0 / h_merge) * (
            self.kernel_height[kernel_min_ind] * self.kernel_center[kernel_min_ind]
            + h_new * s_new
        )
        var_merge = (1.0 / h_merge) * (
            self.kernel_height[kernel_min_ind]
            * (
                np.square(self.kernel_std[kernel_min_ind])
                + np.square(self.kernel_center[kernel_min_ind])
            )
            + h_new * (np.square(std_new) + np.square(s_new))
        ) - np.square(s_merge)

        # Overwrite newly added kernel with properties of the merged one
        self.kernel_height[-1] = h_merge
        self.kernel_center[-1] = s_merge
        self.kernel_std[-1] = np.sqrt(var_merge)

        # Write compressed kernel that was merged and is about to be deleted in delta list with negative height
        self.delta_kernel_height.append(-self.kernel_height[kernel_min_ind])
        self.delta_kernel_center.append(self.kernel_center[kernel_min_ind])
        self.delta_kernel_sigma.append(self.kernel_std[kernel_min_ind])

        # Delete compressed kernel that was merged with new one
        del self.kernel_height[kernel_min_ind]
        del self.kernel_center[kernel_min_ind]
        del self.kernel_std[kernel_min_ind]

        # Count merging events
        # self.merge_count += 1

        return h_merge, s_merge, np.sqrt(var_merge)

    def calc_min_dist(self, cv: np.array):
        """Get minimal Mahalanobis distance to kernels

        Args:
            cv: value to calc distance to

        Returns:
            kernel_min_ind: index of nearest kernel
            min_distance: distance to nearest kernel
        """
        s_diff = cv - np.asarray(self.kernel_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        distance = np.sqrt(
            np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_std))), axis=1)
        )

        kernel_min_ind = np.argmin(distance[:-1]) if len(distance) > 1 else None
        min_distance = distance[kernel_min_ind]

        return kernel_min_ind, min_distance

    def calc_norm_factor(self, approximate: bool = True):
        """Norm factor of probability density

        Returns:
            norm_factor: normalization factor for probability density from kernels
        """
        n_ker = len(self.kernel_center)

        if approximate and n_ker > 10:
            # approximate norm factor to avoid O(N_kernel^2) scaling of exact evaluation
            delta_sum_uprob = 0.0
            for j, s in enumerate(self.delta_kernel_center):

                # Calculate change in probability for changed kernels from delta kernel list
                sign = -1.0 if self.delta_kernel_height[j] < 0 else 1.0
                s_diff = s - np.asarray(self.kernel_center)
                for i in range(self.ncoords):
                    s_diff[:, i] = correct_periodicity(
                        s_diff[:, i], self.periodicity[i]
                    )
                delta_sum_uprob += sign * np.sum(
                    np.asarray(self.kernel_height)
                    * np.exp(
                        -0.5
                        * np.sum(
                            np.square(np.divide(s_diff, np.asarray(self.kernel_std))),
                            axis=1,
                        )
                    )
                )
                delta_sum_uprob += np.sum(
                    np.asarray(self.delta_kernel_height[j])
                    * np.exp(
                        -0.5
                        * np.sum(
                            np.square(
                                np.divide(
                                    s_diff, np.asarray(self.delta_kernel_sigma[j])
                                )
                            ),
                            axis=1,
                        )
                    )
                )
            sum_uprob = (
                self.norm_factor * self.old_KDE_norm * self.old_nker + delta_sum_uprob
            )

        else:
            # analytical calculation of norm factor, inefficient for high number of kernels
            sum_uprob = 0.0
            for s in self.kernel_center:
                sum_gaussians = np.sum(self.calc_gaussians(s))
                sum_uprob += sum_gaussians

        # store KDEnorm and nker for next update
        self.old_KDE_norm = np.copy(self.KDE_norm)
        self.old_nker = n_ker

        return sum_uprob / n_ker / self.KDE_norm

    def grid_potential(self):
        """Calculate bias potential and forces from kernels in bins of `self.grid`

        TODO: update prob_dist and derivative according to `self.delta_kernel_*` lists
              to avoid unfavourable O(N_gridpoints*N_kernels) scaling
        """
        prob_dist = np.zeros_like(self.histogram)
        derivative = np.zeros_like(self.bias)
        if self.ncoords == 1:
            for i, cv in enumerate(self.grid[0]):
                if not self.numerical_forces:
                    val_gaussians = self.calc_gaussians(cv)
                else:
                    val_gaussians, kde_der = self.calc_gaussians(cv, requires_grad=True)
                    derivative[0][0, i] = kde_der
                prob_dist[0, i] = np.sum(val_gaussians)
        else:
            for i, x in enumerate(self.grid[0]):
                for j, y in enumerate(self.grid[1]):
                    if not self.numerical_forces:
                        val_gaussians = self.calc_gaussians(np.asarray([x, y]))
                    else:
                        val_gaussians, kde_der = self.calc_gaussians(
                            np.asarray([x, y]), requires_grad=True
                        )
                        derivative[0][j, i] = kde_der[0]
                        derivative[1][j, i] = kde_der[1]
                    prob_dist[j, i] = np.sum(val_gaussians)

        prob_dist /= self.KDE_norm
        self.bias_potential = self.calc_potential(prob_dist)
        if self.numerical_forces:
            for i in range(self.ncoords):
                self.bias[i] = self.calc_forces(prob_dist, derivative[i])

    def estimate_kernel_std(self, cv):
        """Adaptive estimate of optimal kernel standard deviation from trajectory"""
        self.adaptive_counter += 1
        tau = self.adaptive_std_stride
        if self.adaptive_counter < self.adaptive_std_stride:
            tau = self.adaptive_counter
        self.welford_mean, self.welford_m2, self.welford_var = welford_var(
            self.md_state.step, self.welford_mean, self.welford_m2, cv, tau
        )
        return np.sqrt(self.welford_var)

    def calc_potential(self, prob_dist: float):
        """calc the OPES bias potential from the probability density"""
        return (self.gamma_prefac / self.beta) * np.log(
            prob_dist / self.norm_factor + self.epsilon
        )

    def calc_forces(self, prob_dist: float, deriv_prob_dist: float) -> float:
        """calc the OPES bias forces from the probability density and its derivative"""
        deriv_log = 1.0 / (prob_dist / self.norm_factor + self.epsilon)
        return (
            (self.gamma_prefac / self.beta)
            * deriv_log
            * (deriv_prob_dist / self.norm_factor)
        )
