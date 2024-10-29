import numpy as np
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *
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
        adaptive_std: if adaptive kernel standad deviation is enabled
        adaptive_std_freq: time for estimation of standard deviation in units of `update_freq`
        bandwidth_rescaling: if True, `kernel_std` shrinks during simulation to refine KDE
        explore: enables the OPES explore mode,
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be created
        approximate_norm: enables linear scaling approximation of norm factor
        merge_threshold: threshold distance for kde-merging in units of std, "np.inf" disables merging
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
        kernel_std: np.array = None,
        adaptive_std: bool = True,
        adaptive_std_freq: int = 10,
        bandwidth_rescaling: bool = True,
        explore: bool = False,
        energy_barr: float = 20.0,
        update_freq: int = 100,
        approximate_norm: bool = True,
        merge_threshold: float = np.inf,
        recursive_merge: bool = False,
        bias_factor: float = None,
        numerical_forces: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if kernel_std is None:
            if self.verbose:
                print(
                    f" >>> INFO: estimating kernel standard deviation from initial unbiased MD ({update_freq*adaptive_std_freq} steps)"
                )
        self.adaptive_std = adaptive_std
        self.adaptive_std_stride = adaptive_std_freq * update_freq
        self.adaptive_counter = 0
        self.welford_m2 = np.zeros(self.ncoords)
        self.welford_mean = np.zeros(self.ncoords)
        self.initial_sigma_estimate = False
        if self.adaptive_std or not kernel_std:
            self.sigma_0 = np.full((self.ncoords,), np.nan)
            self.initial_sigma_estimate = True
        elif not hasattr(kernel_std, "__len__"):
            self.sigma_0 = np.asarray([kernel_std])
        else:
            self.sigma_0 = np.asarray(kernel_std)
        self.bandwidth_rescaling = bandwidth_rescaling

        self.explore = explore
        self.update_freq = update_freq
        self.approximate_norm = approximate_norm
        self.merge_threshold = merge_threshold
        self.merge = True
        if self.merge_threshold == np.inf:
            self.merge = False
        self.recursive_merge = recursive_merge
        self.bias_factor = bias_factor
        self.numerical_forces = numerical_forces
        if self.numerical_forces and self.ncoords > 1:
            if self.verbose:
                print(" >>> INFO: Numerical forces disabled as ndim > 1!")
            self.numerical_forces = False

        self.temperature = self.equil_temp
        self.beta = 1.0 / (kB_in_atomic * self.temperature)
        self.energy_barr = energy_barr / atomic_to_kJmol
        self.gamma = (
            self.beta * self.energy_barr if bias_factor == None else bias_factor
        )
        if self.gamma == np.inf:
            self.gamma_prefac = 1.0
        else:
            self.gamma_prefac = self.gamma - 1 if self.explore else 1 - 1 / self.gamma
        self.epsilon = np.exp((-self.beta * self.energy_barr) / self.gamma_prefac)
        self.norm_factor = np.power(self.epsilon, self.gamma_prefac)

        self.n = 0
        self.sum_weights = np.power(self.epsilon, self.gamma_prefac)
        self.sum_weights2 = np.square(self.sum_weights)
        self.n_eff = (
            np.square(1.0 + self.sum_weights) / (1.0 + self.sum_weights2)
            if not self.explore
            else self.n
        )

        self.bias_potential = np.zeros_like(self.histogram)
        self.potential = 0.0

        self.kernel_center = []
        self.kernel_height = []
        self.kernel_std    = []
        self.bias_pot_traj = []

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

        self.md_state = self.the_md.get_sampling_data()
        (cv, grad_cv) = self.get_cv(**kwargs)

        if self.kinetics:
            self._kinetics(grad_cv)

        # get OPES bias force
        forces = self.opes_bias(cv)
        bias_force = self.harmonic_walls(cv, grad_cv)
        for i in range(self.ncoords):
            bias_force += forces[i] * grad_cv[i]
        self.bias_pot_traj.append(self.potential)

        # store biased histogram along CV for output
        if self._check_boundaries(cv):
            bink = self.get_index(cv)
            self.histogram[bink[1], bink[0]] += 1

        # Save values for traj output
        if traj_file:
            self.traj = np.append(self.traj, [cv], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)

        # Write output
        if self.md_state.step % self.out_freq == 0:
            if traj_file:
                self.write_traj(filename=traj_file)
            if out_file:
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
        if self.ncoords == 1:
            prob_dist = np.zeros_like(self.grid)
            for i, cv in enumerate(self.grid[0]):
                val_gaussians = self.calc_gaussians(cv)
                prob_dist[0, i] = np.sum(val_gaussians)
        else:
            return np.zeros_like(self.grid)

        self.bias_potential = self.calc_potential(prob_dist)
        pmf = -self.bias_potential / self.gamma_prefac
        pmf -= pmf.min()
        return pmf

    def write_traj(self, filename="CV_traj.dat"):
        data = {}
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp
        data["Bias pot"] = self.bias_pot_traj

        self._write_traj(data, filename=filename)

        # Reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.bias_pot_traj = [self.bias_pot_traj[-1]]

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
        # kernel `sigma_0` estimate from MD
        if (
            self.initial_sigma_estimate
            and self.md_state.step < self.adaptive_std_stride
        ):
            self.estimate_kernel_std(cv)
            return np.zeros_like(self.the_md.coords)

        elif self.initial_sigma_estimate or self.adaptive_std:
            self.sigma_0 = self.estimate_kernel_std(cv)
            if self.initial_sigma_estimate and self.verbose:
                print(
                    f" >>> INFO: kernel standard deviation for OPES is set to {self.sigma_0}"
                )
            self.initial_sigma_estimate = False

        # OPES KDE update
        if self.md_state.step % self.update_freq == 0:
            self.update_kde(cv)
            # if self.verbose:
            #    print(f" >>> INFO: OPES KDE updated, N_kernels={len(self.kernel_height)}")

        if self.numerical_forces and self._check_boundaries(cv):
            # read numerical bias potential and forces from grid
            idx = self.get_index(cv)
            self.potential = self.bias_potential[idx[0]]
            return [self.forces_numerical[idx[0]]]
        else:
            # calc analytical bias potential and forces from KDE
            gaussians, kde_derivative = self.calc_gaussians(cv, requires_grad=True)
            self.prob_dist = np.sum(gaussians)
            self.potential = self.calc_potential(self.prob_dist)
            return self.calculate_forces(self.prob_dist, kde_derivative)

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
            return np.zeros(1)

        KDE_norm = self.sum_weights if not self.explore else self.n

        # distance to kernel centers
        s_diff = cv - np.asarray(self.kernel_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        # evaluate values of kernels at cv
        gaussians = (
            np.asarray(self.kernel_height)
            / KDE_norm
            * np.exp(
                -0.5
                * np.sum(
                    np.square(np.divide(s_diff, np.asarray(self.kernel_std))), axis=1
                )
            )
        )
        if requires_grad:
            kde_derivative = np.sum(
                -gaussians
                * np.divide(s_diff, np.square(np.asarray(self.kernel_std))).T,
                axis=1,
            )
            return gaussians, kde_derivative

        return gaussians

    def update_kde(self, cv):
        """on-the-fly update of kernel density estimation of probability density along CVs

        Args:
            CV: new value of CVS
        """
        self.delta_kernel_height = []
        self.delta_kernel_center = []
        self.delta_kernel_sigma = []

        self.n += 1

        # Calculate probability density at CV
        self.prob_dist = np.sum(self.calc_gaussians(cv))

        # Calculate bias potential
        self.potential = self.calc_potential(self.prob_dist)

        # Calculate weight coefficients
        weight_coeff = np.exp(self.beta * self.potential)
        self.sum_weights += weight_coeff
        self.sum_weights2 += weight_coeff * weight_coeff

        # Bandwidth rescaling
        self.n_eff = (
            np.square(1.0 + self.sum_weights) / (1.0 + self.sum_weights2)
            if not self.explore
            else self.n
        )

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
        self.norm_factor = self.calc_norm_factor(approximate=self.approximate_norm)
        self.pmf = self.get_pmf()
        if self.numerical_forces:
            self.forces_numerical = np.gradient(self.bias_potential, self.grid[0])

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

    def calc_norm_factor(self, approximate: bool=True):
        """Norm factor of probability density (configurational integral)

        Returns:
            norm_factor: normalization factor for probability density from kernels
        """
        N = len(self.kernel_center)
        KDE_norm = self.sum_weights if not self.explore else self.n

        if approximate and N > 10:
            # approximate norm factor to avoid O(N_kernel^2) scaling of exact evaluation
            delta_uprob = 0.0
            for j, s in enumerate(self.delta_kernel_center):

                # Calculate change in probability for changed kernels from delta kernel list
                sign = -1.0 if self.delta_kernel_height[j] < 0 else 1.0
                s_diff = s - np.asarray(self.kernel_center)
                for i in range(self.ncoords):
                    s_diff[:, i] = correct_periodicity(
                        s_diff[:, i], self.periodicity[i]
                    )
                delta_sum_uprob = sign * np.sum(
                    np.asarray(self.kernel_height)
                    / KDE_norm
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
                    / KDE_norm
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
                delta_uprob += delta_sum_uprob

            new_uprob = self.uprob_old + delta_uprob
            self.uprob_old = new_uprob
            norm_factor = new_uprob / N / KDE_norm

        else:
            # analytical calculation of norm factor, inefficient for high number of kernels
            uprob = 0.0
            for s in self.kernel_center:
                sum_gaussians = np.sum(self.calc_gaussians(s))
                uprob += sum_gaussians
            self.uprob_old = uprob
            norm_factor = uprob / N / KDE_norm 
        return norm_factor

    def estimate_kernel_std(self, cv):
        """Adaptive estimate of optimal kernel standard deviation from trajectory"""
        self.adaptive_counter += 1
        tau = self.adaptive_std_stride
        if self.adaptive_counter < self.adaptive_std_stride:
            tau = self.adaptive_counter
        self.welford_mean, self.welford_m2, self.welford_var = welford_var(
            self.md_state.step, self.welford_mean, self.welford_m2, cv, tau
        )
        return (
            np.sqrt(self.welford_var / self.gamma)
            if not self.explore
            else np.sqrt(self.welford_var)
        )

    def calc_potential(self, prob_dist: float):
        """calc the OPES bias potential from the probability density"""
        potential = (self.gamma_prefac / self.beta) * np.log(
            prob_dist / self.norm_factor + self.epsilon
        )
        return potential

    def calculate_forces(self, prob_dist: float, deriv_prob_dist: float) -> float:
        """calc the OPES bias forces from the probability density and its derivative"""
        deriv_log = 1.0 / (prob_dist + self.norm_factor * self.epsilon)
        deriv_pot = (self.gamma_prefac / self.beta) * deriv_log * deriv_prob_dist
        return deriv_pot
