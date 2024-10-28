import torch 
import numpy as np
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *
from .utils import correct_periodicity
from .utils import welford_var

class OPES_MINIMAL(EnhancedSampling):

    def __init__(
        self,
        *args,
        kernel_std: torch.tensor = None,
        explore: bool=False,
        energy_barr: float = 20.0,
        update_freq: int = 1000,
        approximate_norm: bool = True,
        exact_norm: bool = False,
        merge_threshold: float = 1.0,
        recursive_merge: bool = False,
        bias_factor: float = None,
        numerical_forces: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if kernel_std is None:
            raise ValueError(" >>> OPES: invalid kernel_std")
     
        self.sigma_0 = kernel_std
        self.explore = explore
        self.update_freq = update_freq
        self.approximate_norm = approximate_norm
        self.exact_norm = exact_norm
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
        self.beta = 1. / (kB_in_atomic * self.temperature)
        self.energy_barr = energy_barr / atomic_to_kJmol
        self.gamma = (
            self.beta * self.energy_barr if bias_factor == None else bias_factor
        )  
        if self.gamma == np.inf:
            self.gamma_prefac = 1.0
        else:
            self.gamma_prefac = (
                self.gamma - 1 if self.explore else 1 - 1 / self.gamma
            ) 
        self.epsilon = np.exp((-self.beta * self.energy_barr) / self.gamma_prefac)  
        self.norm_factor = np.power(self.epsilon, self.gamma_prefac)

        self.n = 0
        self.sum_weights = np.power(self.epsilon, self.gamma_prefac)
        self.sum_weights2 = np.square(self.sum_weights)
        self.n_eff = (
            np.square(1. + self.sum_weights) / (1. + self.sum_weights2)
            if not self.explore 
            else self.n           
        )

        self.kernel_center = []
        self.kernel_height = []
        self.kernel_std    = []
        self.bias_pot_traj = []

    def step_bias(
        self, 
        traj_file: str="CV_traj.dat", 
        **kwargs
    ) -> np.array:
        """Apply OPES bias to MD

        Returns:
            bias_force: Bias force that has to be added to system forces
        """

        self.md_state = self.the_md.get_sampling_data()
        (s_new, delta_s_new) = self.get_cv(**kwargs)

        if self.kinetics:
            self._kinetics(delta_s_new)

        forces = self.opes_bias(s_new)
        self.bias_pot_traj.append(self.potential)

        bias_force = self.harmonic_walls(s_new, delta_s_new)  # , 1.6 * self.hill_std)
        for i in range(self.ncoords):
            bias_force += forces[i] * delta_s_new[i]

        # Save values for traj
        self.traj = np.append(self.traj, [s_new], axis=0)
        self.epot.append(self.md_state.epot)
        self.temp.append(self.md_state.temp)

        # Write output
        if self.md_state.step % self.out_freq == 0:

            self.write_traj(filename=traj_file)
            self.write_restart()

        return bias_force

    def get_pmf(self) -> np.array:
        """Calculate current PMF estimate on `self.grid`

        Returns:
            pmf: current PMF estimate from OPES kernels
        """
        if self.ncoords == 1:
            P = np.zeros_like(self.grid[0])
            for x in range(len(self.grid[0])):
                val_gaussians = self.calc_gaussians(self.grid[0][x])
                P[x] = np.sum(val_gaussians) 

        elif self.ncoords == 2:
            P = np.zeros_like(self.grid)
            KDE_norm = self.sum_weights if not self.explore else self.n
            for x in range(len(self.grid[0, :])):
                for y in range(len(self.grid[1, :])):
                    s_diff = np.array([self.grid[0, x], self.grid[1, y]]) - np.asarray(
                        self.kernel_center
                    )
                    for l in range(self.ncoords):
                        s_diff[:, l] = correct_periodicity(
                            s_diff[:, l], self.periodicity[l]
                        )
                    val_gaussians = np.asarray(self.kernel_height) * np.exp(
                        -0.5
                        * np.sum(
                            np.square(np.divide(s_diff, np.asarray(self.kernel_std))),
                            axis=1,
                        )
                    )
                    P[x, y] = np.sum(val_gaussians) / KDE_norm
        else:
            pmf = np.zeros_like(self.grid)
            return pmf
        
        P /= self.norm_factor
        self.bias_potential = (np.log(P + self.epsilon) / self.beta) * self.gamma_prefac
        pmf = -self.bias_potential / self.gamma_prefac
        
        pmf -= pmf.min()
        return pmf

    def write_traj(self, filename='CV_traj.dat'):
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
        raise ValueError(" >>> ERROR: Multiple-walker shared bias not available for OPES!")

    def write_restart(self, filename: str="restart_opes"):
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

    def restart(self, filename: str="restart_opes"):
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
            bias force: len(ncoords) array of bias forces on CVs
        """
        if self.md_state.step % self.update_freq == 0:
            self.update_kde(cv)

        if self.numerical_forces and self._check_boundaries(cv):
            # read numerical bias potential and forces from grid
            idx = self.get_index(cv)
            self.potential = self.bias_potential[idx[0]]
            return [self.forces_numerical[idx[0]]]
        else:
            # calc analytical bias potential and forces from KDE at cv
            kde, kde_derivative = self.calc_gaussians(cv, requires_grad=True)
            self.prob_dist = np.sum(kde) 
            self.potential = self.calc_potential(self.prob_dist)
            return self.calculate_forces(self.prob_dist, kde_derivative)

    def calc_gaussians(self, cv, requires_grad: bool=False) -> np.array:
        """Get current value of KDE from Gaussian hills

        Args:
            cv: value of CV where the KDE should be evaluated
            requires_grad: if True, gradient of KDE is returned as second argument
        """

        if len(self.kernel_center) == 0:
            return np.zeros(1)

        KDE_norm = self.sum_weights if not self.explore else self.n

        # distance to kernel centers
        s_diff = cv - np.asarray(self.kernel_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        # Calculate values of Gaussians at center of kernel currently in loop and sum them
        kde = np.asarray(self.kernel_height) / KDE_norm * np.exp(-0.5 * np.sum(
            np.square(np.divide(s_diff, np.asarray(self.kernel_std))), axis=1)
        ) 
        if requires_grad:
            kde_derivative = np.sum(-kde * np.divide(s_diff, np.square(np.asarray(self.kernel_std))).T, axis=1,)
            return kde, kde_derivative

        return kde

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
            np.square(1. + self.sum_weights) / (1. + self.sum_weights2)
            if not self.explore 
            else self.n           
        )

        if len(self.kernel_std) > 0:
            sigma_i = self.sigma_0 * np.power(
                (self.n_eff * (self.ncoords + 2.) / 4.), -1. / (self.ncoords + 4.)
            )
        else: 
            sigma_i = self.sigma_0

        height = (
            weight_coeff * np.prod(self.sigma_0 / sigma_i) if not self.explore else 1.0
        )

        # Kernel Density
        self.add_kernel(height, cv, sigma_i)

        # Calculate normalization factor
        self.norm_factor = self.calc_norm_factor()
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

        # Append final merged kernel or if no merging occurred just the new kernel to delta list
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
        #self.merge_count += 1

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
    
    def calc_norm_factor(self):
        """Norm factor of probability density (configurational integral)

        Returns:
            norm_factor: normalization factor for probability density from kernels
        """
        N = len(self.kernel_center)
        KDE_norm = self.sum_weights if not self.explore else self.n

        if self.approximate_norm and N > 10:
            delta_uprob = 0.0
            for j, s in enumerate(self.delta_kernel_center):

                KDE_norm = self.sum_weights

                # Calculate change in probability for changed kernels by delta kernel list
                sign = -1.0 if self.delta_kernel_height[j] < 0 else 1.0
                s_diff = s - np.asarray(self.kernel_center)
                for i in range(self.ncoords):
                    s_diff[:, i] = correct_periodicity(
                        s_diff[:, i], self.periodicity[i]
                    )
                delta_sum_uprob = sign * np.sum(
                    np.asarray(self.kernel_height) / KDE_norm
                    * np.exp(
                        -0.5
                        * np.sum(
                            np.square(np.divide(s_diff, np.asarray(self.kernel_std))),
                            axis=1,
                        )
                    )
                )
                delta_sum_uprob += np.sum(
                    np.asarray(self.delta_kernel_height[j]) / KDE_norm
                    * np.exp(
                        -0.5
                        * np.sum(
                            np.square(np.divide(s_diff, np.asarray(self.delta_kernel_sigma[j]))),
                            axis=1,
                        )
                    )
                )
                delta_uprob += delta_sum_uprob

            new_uprob = self.uprob_old + delta_uprob
            self.uprob_old = new_uprob
            norm_factor = new_uprob / N / KDE_norm

        else:
            # exact norm factor, slow in long run as O(N_kernels^2)!
            uprob = 0.0
            for s in self.kernel_center:
                sum_gaussians = np.sum(self.calc_gaussians(s))
                uprob += sum_gaussians
            self.uprob_old = uprob 
            norm_factor = uprob / N / KDE_norm
        return norm_factor
    
    def calc_potential(self, prob_dist: float):
        """calc the OPES bias potential from the probability density"""
        potential = (self.gamma_prefac / self.beta) * np.log(
            prob_dist / self.norm_factor + self.epsilon
        )
        return potential

    def calculate_forces(self, prob_dist: float, deriv_prob_dist: float) -> float:
        """calc the OPES bias forces from the probability density and its derivative"""
        deriv_log = 1. / (prob_dist + self.norm_factor * self.epsilon)
        deriv_pot = (
            (self.gamma_prefac / self.beta) * deriv_log * deriv_prob_dist
        )
        return deriv_pot   
    