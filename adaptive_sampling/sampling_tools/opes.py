import sys,os
import numpy as np
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *
from .utils import correct_periodicity
from .utils import welford_var

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:
        kernel_std: standard deviation of first kernel
        explore: enables exploration mode
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be 
        approximate_norm: enables approximation of norm factor
        exact_norm: enables exact calculation of norm factor, if both are enabled, exact is used every 100 updates
        merge_threshold: threshold distance for kde-merging in units of std, "np.inf" disables merging
        recursion_merge: enables recursive merging
        bias_factor: allows setting a default bias factor instead of calculating it from energy barrier
        print_pmf: enables calculation of pmf on the fly
        adaptive_sigma: enables adaptive sigma calculation nontheless with rescaling
        unbiased_time: time in update frequencies for unbiased estimation of sigma if no input is given
        fixed_sigma: disables bandwidth rescaling and uses input sigma for all kernels
    """
    def __init__(
        self,
        *args,
        kernel_std: np.array = np.array([None, None]),
        explore: bool = False,
        energy_barr: float = 20.0,
        update_freq: int = 1000,
        approximate_norm: bool = True,
        exact_norm: bool = False,
        merge_threshold: float = 1.0,
        recursion_merge: bool = False,
        bias_factor: float = None,
        print_pmf: bool = False,
        adaptive_sigma: bool = False,
        unbiased_time: int = 10,
        fixed_sigma: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Simulation Parameters
        self.explore = explore
        self.update_freq = update_freq
        self.approx_norm = approximate_norm
        self.exact_norm = exact_norm
        if not self.approx_norm and not self.exact_norm:
            raise ValueError(" >>> Error: Either approximate or exact norm factor calculation must be enabled!")
        self.adaptive_sigma = adaptive_sigma
        self.fixed_sigma = fixed_sigma
        if self.fixed_sigma and self.adaptive_sigma:
            raise ValueError(" >>> Error: Adaptive and fixed sigma cannot be enabled at the same time!")
        self.merge = False if merge_threshold == np.inf else True
        self.merge_threshold = merge_threshold
        self.recursive = recursion_merge
        s, _ = self.get_cv(**kwargs)
        self.print_pmf = print_pmf
        self.sigma_estimate = True if (kernel_std == None).all() else False
        if (kernel_std == None).any() and not (kernel_std == None).all():
            raise ValueError(" >>> Error: Kernel standard deviation must be set either all None or all values!")

        # Constants
        self.energy_barr = energy_barr * kJ_to_kcal
        self.beta = 1/(self.equil_temp * kB_in_atomic * atomic_to_kJmol * kJ_to_kcal)
        self.gamma = self.beta * self.energy_barr if bias_factor == None else bias_factor # Enable setting of bias factor
        self.gamma_prefac = self.gamma - 1 if explore else 1 - 1/self.gamma # Differentiate between explore and normal opes
        self.temperature = self.equil_temp
        self.epsilon = np.exp((-self.beta * self.energy_barr) / self.gamma_prefac)

        # Initial values
        self.prob_dist = 1.0
        self.deriv_prob_dist = 0.0
        self.potential = 0.0
        self.sum_weights = np.power(self.epsilon, self.gamma_prefac)
        self.sum_weights_square = self.sum_weights * self.sum_weights
        self.norm_factor = 1/self.sum_weights
        self.uprob_old = self.norm_factor * self.sum_weights
        self.md_state = self.the_md.get_sampling_data()
        if (kernel_std != None).all():
            self.sigma_0 = self.unit_conversion_cv(np.asarray(kernel_std))[0] # Standard deviation of first kernel converted to atomic units
        self.n = 0 # Counter for number of updates
        #self.pmf = 0.0
        self.adaptive_sigma_stride = unbiased_time * self.update_freq
        self.adaptive_counter = 0
        self.welford_m2 = np.zeros(self.ncoords)
        self.welford_mean = np.zeros(self.ncoords)

        # Kernels
        self.kernel_height = []
        self.kernel_center = []
        self.kernel_sigma = []

        # Output
        self.output_sum_weigths = []
        self.output_sum_weigths_square = []
        self.output_norm_factor = []
        self.output_bias_pot = []
        self.merge_count = 0
        self.uprob_print = self.uprob_old
        self.n_eff = 1.0

        # Error catching
        if self.kinetics == True and self.ncoords > 1:
            raise ValueError(" >>> Error: Kinetics can only be calculated for one-dimensional CV space!")


    def step_bias(
        self, 
        write_output: bool = True, 
        write_traj: bool = True, 
        output_file: str = 'opes.out',
        traj_file: str = 'CV_traj.dat', 
        restart_file: str = 'restart_opes',
        **kwargs
    ):
        """Applies OPES to MD: pulls sampling data and CV, drops a gaussian every Udate_freq MD-steps, calculates the bias force

        Args:
            write_output: enables output file
            write_traj: enables traj file
        
        Returns:
            bias_force: bias_force on location in CV space in atomic units
        """
        # Load md data
        self.md_state = self.the_md.get_sampling_data()
        (s_new, delta_s_new) = self.get_cv(**kwargs)

        # Corretion for kinetics
        if self.kinetics:
            self._kinetics(delta_s_new)

        # Calculate derivative of potential
        forces = self.opes_bias(s_new)

        # Calculate bias force
        bias_force = np.zeros_like(self.the_md.coords, dtype=float)
        for i in range(self.ncoords):
            bias_force += forces[i] * delta_s_new[i]
        bias_force = bias_force / kJ_to_kcal / atomic_to_kJmol

        bias_force += self.harmonic_walls(s_new, delta_s_new)  # , 1.6 * self.hill_std)

        # Save values for traj
        self.traj = np.append(self.traj, [s_new], axis=0)
        self.epot.append(self.md_state.epot)
        self.temp.append(self.md_state.temp)
        self.output_bias_pot.append(self.potential)
        self.output_sum_weigths.append(self.sum_weights)
        self.output_sum_weigths_square.append(self.sum_weights_square)
        self.output_norm_factor.append(self.norm_factor)

        # Write output
        if self.md_state.step % self.out_freq == 0:

            if write_traj:
                self.write_traj(filename=traj_file)

            if write_output:
                #self.write_output(filename=output_file)
                self.write_restart(filename=restart_file)

        return bias_force
    

    def opes_bias(
        self,
        s_new: np.array
    ) -> np.array:
        """calculate the bias force for a given location in CV space by the OPES algorithm e.g. for extended system usecase
        
        Args:
            s_new: location in CV space for which the bias force is wanted, for extended system its cv position of fictional particle

        Returns:
            bias_force: bias force on location in CV space in atomic units
        """
        if self.verbose and self.md_state.step%self.update_freq == 0:
            print("OPES bias called at ", s_new)

        # Unbiased estimation of sigma
        if self.sigma_estimate and self.md_state.step < self.adaptive_sigma_stride:
            self.adaptive_counter += 1
            tau = self.adaptive_sigma_stride
            if self.adaptive_counter < self.adaptive_sigma_stride:
                tau = self.adaptive_counter
            self.welford_mean, self.welford_m2, self.welford_var = welford_var(self.md_state.step, self.welford_mean, self.welford_m2, s_new, tau)
            self.sigma_0 = np.sqrt(self.welford_var)

            self.traj = np.append(self.traj, [s_new], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)
            self.output_bias_pot.append(0.0)
            self.output_sum_weigths.append(self.sum_weights)
            self.output_sum_weigths_square.append(self.sum_weights_square)
            self.output_norm_factor.append(self.norm_factor)
            return np.zeros_like(self.the_md.coords)

        # Update sigma if adaptive sigma is enabled
        if self.adaptive_sigma:
            self.adaptive_counter += 1
            tau = self.adaptive_sigma_stride
            if self.adaptive_counter < self.adaptive_sigma_stride:
                tau = self.adaptive_counter
            self.welford_mean, self.welford_m2, self.welford_var = welford_var(self.md_state.step, self.welford_mean, self.welford_m2, s_new, tau)
            factor = self.gamma if not self.explore else 1.0
            self.sigma_0 = np.sqrt(self.welford_var / factor)
        
        # Call update function to place kernel
        if self.md_state.step%self.update_freq == 0:
            if self.verbose:
                print("OPES update KDE started.")
            self.update_kde(s_new)

        # Calculate new probability and its derivative
        KDE_norm = self.sum_weights if not self.explore else self.n
        val_gaussians = self.get_val_gaussian(s_new)
        prob_dist = np.sum(val_gaussians) / KDE_norm

        s_diff = (s_new - np.asarray(self.kernel_center))
        for i in range(self.ncoords):
            s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])

        deriv_prob_dist = np.sum(-val_gaussians * np.divide(s_diff, np.asarray(self.kernel_sigma)).T, axis=1) / KDE_norm

        self.prob_dist = prob_dist
        self.deriv_prob_dist = deriv_prob_dist
        if self.verbose and self.md_state.step%self.update_freq==0:
            print("Probabiliy distribution: ", prob_dist, "and its derivative: ", deriv_prob_dist)
            print("val gaussians =  ", val_gaussians)

        # Calculate Potential
        self.potential = self.calc_pot(prob_dist)

        # Calculate forces
        forces = self.calculate_forces(prob_dist, deriv_prob_dist)
        
        return forces
    

    def update_kde(
        self, 
        s_new: np.array
    ):
        """main function in algorithm; calls compression and calculates weigths
        
        Args:
            s_new: center of new gaussian

        """
        self.delta_kernel_height = []
        self.delta_kernel_center = []
        self.delta_kernel_sigma = []

        self.n += 1
        KDE_norm = self.sum_weights if not self.explore else self.n

        # Calculate probability
        prob_dist = np.sum(self.get_val_gaussian(s_new)) / KDE_norm
        self.prob_dist = prob_dist

        # Calculate bias potential
        potential = self.calc_pot(prob_dist)

        # Calculate weight coefficients
        weigth_coeff = np.exp(self.beta * potential)
        if self.verbose:
            print("Update function called, now evaluating its properties")
            #print("Probability distribution: ", prob_dist)
            #print("potential is: ", potential)
        self.sum_weights += weigth_coeff
        self.sum_weights_square += weigth_coeff * weigth_coeff

        # Bandwidth rescaling
        self.n_eff = np.square(1+self.sum_weights) / (1+self.sum_weights_square) if not self.explore else self.n

        if not self.fixed_sigma and len(self.kernel_sigma) > 0:
            sigma_i = self.sigma_0 * np.power((self.n_eff * (self.ncoords + 2)/4), -1/(self.ncoords + 4))
        else:
            sigma_i = self.sigma_0
        
        height = weigth_coeff * np.prod(self.sigma_0 / sigma_i) if not self.explore else 1.0

        # Kernel Density 
        self.compression_check(height, s_new, sigma_i)

        # Calculate normalization factor
        if self.exact_norm and self.approx_norm:
            if self.n % 100 == 0:
                self.norm_factor = self.calc_norm_factor(approximate = False)
            self.norm_factor = self.calc_norm_factor(approximate = True)
        else:
            self.norm_factor = self.calc_norm_factor(approximate = self.approx_norm)


        # Calculate pmf on the fly if enabled
        if self.print_pmf:
            self.pmf = self.get_pmf()


    def get_val_gaussian(
        self,
        s: np.array,
    ) -> np.array: 
        
        """get the values of all gaussians at point s_new
        
        Args:
            s = point of interest
            
        Returns:
            val_gaussians: array of all values of all kernels at s
        """
        if len(self.kernel_center) == 0:
            return np.zeros(1)

        s_diff = (s - np.asarray(self.kernel_center))

        # Correct Periodicity of spatial distances
        for i in range(self.ncoords):
            s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])
            
        # Calculate values of Gaussians at center of kernel currently in loop and sum them
        if False and self.verbose and self.md_state.step%self.update_freq ==0 and self.md_state.step > 0:
            print("S_diff", s_diff)
            print(np.asarray(self.kernel_sigma))
        val_gaussians = np.asarray(self.kernel_height) * \
            np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))

        return val_gaussians


    def calc_pot(
        self, 
        prob_dist: float
    ):
        """calculate the potential of given location
        
        Args:
            prob_dist: local probability distribution on which potential shall be calculated

        Returns:
            potential: potential for given probability distribution
        """
        potential = (self.gamma_prefac / self.beta) * np.log(prob_dist / self.norm_factor + self.epsilon)
        return potential
    
    
    def calculate_forces(
        self, 
        prob_dist:float, 
        deriv_prob_dist: float
    ) -> float:
        """calculate the forces as derivative of potential

        Args:
            prob_dist: probability disitrbution on wanted location and its derivative 

        Returns:
            deriv_pot: derivative of potential for location s
        """
        deriv_pot = (self.gamma_prefac/self.beta) *\
              (1/ ((prob_dist / self.norm_factor) + self.epsilon)) * (deriv_prob_dist / self.norm_factor)
        if self.verbose and self.md_state.step%self.update_freq==0:
            print("derivate of pot: ", deriv_pot)

        return deriv_pot


    def calc_norm_factor(
        self,
        approximate: bool = True
    ):
        """approximatec the norm factor with respect to existing gaussians by adding the change for newly added kernel

        Args:
            approx_for_loop: enable double-for loop for approximation for testin
            approximate: enable normalization factor approximation with delta_kernel lists

        Returns:
            delta_uprob: non normalized sum over the gauss values for the newly placed kernel
        """
        S = self.sum_weights if not self.explore else self.n
        N = len(self.kernel_center)
        
        if approximate:
            delta_uprob =0.0
            for j, s in enumerate(self.delta_kernel_center):

                # Sign for correct probability correction from deleted, merged or added kernels
                sign = -1.0 if self.delta_kernel_height[j] < 0 else 1.0

                # Calculate spatial distances and correct periodicity
                s_diff = (s - np.asarray(self.kernel_center))
                for i in range(self.ncoords):
                    s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])

                # Calculate change in probability for changed kernels by delta kernel list
                delta_sum_uprob = sign * np.sum(np.asarray(self.kernel_height) *\
                        np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1)))
                delta_sum_uprob += np.sum(np.asarray(self.delta_kernel_height[j]) *\
                        np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.delta_kernel_sigma[j]))),axis=1)))
                delta_uprob += delta_sum_uprob

            # Get old uprob from denormalized old norm factor and add change in uprob then calculate new norm factor and set
            new_uprob = self.uprob_old + delta_uprob
            self.uprob_old = new_uprob
            self.uprob_print = new_uprob
            norm_factor = new_uprob/N/S

            return norm_factor

        else:
            uprob = 0.0
            # Loop over all kernels with all kernels to get exact uprob
            for s in self.kernel_center:
                sum_gaussians =np.sum(self.get_val_gaussian(s))
                uprob += sum_gaussians

            # Get norm factor from exact uprob and set it
            self.uprob_old = uprob
            norm_factor = uprob/N/S
            return norm_factor


    def compression_check(
        self, 
        h_new: float, 
        s_new: np.array, 
        std_new: np.array
    ):
        """kde compression check: new kernel added in update function is tested for being to near to an existing compressed one;
         if so, merge kernels and delete added, check recursive, otherwise skip

        Args:
            heigth: weighted heigth of new kernel
            s_new: center of new kernel
            std_new: stndard deviation of new kernel
        """
        # Set up and update kernel lists

        if self.verbose:
            print("Kernels before adding:")
            self.show_kernels()
        self.kernel_height.append(h_new)
        self.kernel_center.append(s_new)
        self.kernel_sigma.append(std_new)
        if self.verbose:
            print("Kernels after adding:")
            self.show_kernels()

        kernel_min_ind, min_distance = self.calc_min_dist(s_new)

        if self.verbose:
            print("Kernel added: ", h_new, self.kernel_center[-1], std_new)
            print("min distance: ", min_distance, " to sampling point: ", s_new)

        # Recursive merging if enabled and distances under threshold
        while self.merge and np.all(min_distance < self.merge_threshold) and len(self.kernel_center) > 1:
            if self.verbose:
                print("Merging started.")
                print("Kernel to merge: ", self.kernel_height[kernel_min_ind], self.kernel_center[kernel_min_ind], self.kernel_sigma[kernel_min_ind])
            # Merge again
            h_new, s_new, std_new = self.merge_kernels(kernel_min_ind, h_new, s_new, std_new)

            # Calculate new distances to update while condition
            kernel_min_ind, min_distance = self.calc_min_dist(s_new)

            if not self.recursive:
                break

        # Append final merged kernel or if no merging occured just the new kernel to delta list
        self.delta_kernel_height.append(h_new)
        self.delta_kernel_center.append(s_new)
        self.delta_kernel_sigma.append(std_new)
            

    def calc_min_dist(
        self,
        s_new: np.array
    ):
        """calculate distances to all compressed kernels and get the minimal distance as well as the corresponding kernel

        Args:
        s_new: center of new kernel for which the distances are needed

        Returns:
        kernel_min_ind: index of kernel in self.kernel lists that is nearest to new one
        min_distance: distance to metioned kernel
        """
        # Calculate spatial distances of sampling point and all kernel centers
        s_diff = s_new - np.asarray(self.kernel_center)

        # Correct Periodicity of spatial distances
        for i in range(self.ncoords):
            s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])

        distance = np.sqrt(np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))), axis=1))
        if self.verbose:
            print("distances: ", distance)
        kernel_min_ind = np.argmin(distance[:-1]) if len(distance) > 1 else None
        min_distance = distance[kernel_min_ind]

        return kernel_min_ind, min_distance


    def merge_kernels(
        self, 
        kernel_min_ind: int, 
        h_new: float, 
        s_new: np.array, 
        std_new: np.array
    ) -> list:
        """merge two kernels calculating the characteristics of a new one with respect to the old ones and overwrite old one with merged

        Args:
            kernel_mind_ind: index of nearest kernel, which is to be merged
            h_new: height of new kernel
            s_new: center of new kernel
            std_new: standard deviation of new kernel

        Returns:
            height, center and standard deviation of merged kernel       
        """
        # Calculate properties of merged kernel
        h_merge = self.kernel_height[kernel_min_ind] + h_new
        s_merge = (1.0/h_merge)*(self.kernel_height[kernel_min_ind] * self.kernel_center[kernel_min_ind] + h_new * s_new)
        var_merge = (1.0/h_merge)*(self.kernel_height[kernel_min_ind] * (np.square(self.kernel_sigma[kernel_min_ind]) +\
                 np.square(self.kernel_center[kernel_min_ind])) + h_new * (np.square(std_new) + np.square(s_new))) - np.square(s_merge)
        
        # Overwrite newly added kernel with properties of the merged one
        self.kernel_height[-1] = h_merge
        self.kernel_center[-1] = s_merge
        self.kernel_sigma[-1] = np.sqrt(var_merge)

        # Write compressed kernel that was merged and is about to be deleted in delta list with negative height
        self.delta_kernel_height.append(-self.kernel_height[kernel_min_ind])
        self.delta_kernel_center.append(self.kernel_center[kernel_min_ind])
        self.delta_kernel_sigma.append(self.kernel_sigma[kernel_min_ind])

        # Delete compressed kernel that was merged with new one
        del self.kernel_height[kernel_min_ind]
        del self.kernel_center[kernel_min_ind]
        del self.kernel_sigma[kernel_min_ind]

        # Count merging events
        self.merge_count += 1

        return h_merge, s_merge, np.sqrt(var_merge)


    def write_restart(
        self, 
        filename: str="restart_opes"
    ):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            sum_weigths=self.sum_weights,
            sum_weigths_square = self.sum_weights_square,
            norm_factor = self.norm_factor,
            heigth = self.kernel_height,
            center = self.kernel_center,
            sigma = self.kernel_sigma,
            explore = self.explore,
            n = self.n
        )


    def restart(
        self, 
        filename: str = "restart_opes"
    ):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        # Load dicitionary entries from restart file
        self.sum_weights = float(data["sum_weigths"])
        self.sum_weights_square = float(data["sum_weigths_square"])
        self.norm_factor = float(data["norm_factor"])
        self.kernel_height = data["heigth"]
        self.kernel_center = data["center"]
        self.kernel_sigma = data["sigma"]
        self.explore = data["explore"]
        self.n = data["n"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")


    def write_traj(
        self, 
        filename: str = 'CV_traj.dat'
    ):
        """save trajectory for post-processing
        """
        data = {}
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp
        data["Bias pot"] = self.output_bias_pot

        self._write_traj(data, filename=filename)

        # Reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.output_bias_pot = [self.output_bias_pot[-1]]


    def show_kernels(
        self
    ):
        """for testing print kernels
        """
        if self.verbose: #and self.md_state.step%(self.update_freq*100)==0:
            print("Kernels: ")
            print("heights ", self.kernel_height)
            print("centers ", self.kernel_center)
            print("sigmas ", self.kernel_sigma)
        else:
            pass
        
    
    def get_pmf(
        self
    ):
        """calculate pmf on the fly from compressed kernels for one and two dimensional CV spaces

        Returns:
            pmf: potential of mean force in kcal
        """
        if self.ncoords == 1:
            P = np.zeros_like(self.grid[0])
            KDE_norm = self.sum_weights if not self.explore else self.n
            for x in range(len(self.grid[0])):
                s_diff = self.grid[0][x] - np.asarray(self.kernel_center)
                for l in range(self.ncoords):
                    s_diff[:,l] = correct_periodicity(s_diff[:,l], self.periodicity[l])
                val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
                P[x] = np.sum(val_gaussians) / KDE_norm

        elif self.ncoords == 2:
            P = np.zeros_like(self.grid)
            KDE_norm = self.sum_weights if not self.explore else self.n
            for x in range(len(self.grid[0,:])):
                for y in range(len(self.grid[1,:])):
                    s_diff = np.array([self.grid[0,x], self.grid[1,y]]) - np.asarray(self.kernel_center)
                    for l in range(self.ncoords):
                        s_diff[:,l] = correct_periodicity(s_diff[:,l], self.periodicity[l])
                    val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
                    P[x,y] = np.sum(val_gaussians) / KDE_norm
        else:
            pmf = 0.0
            return pmf
        
        bias_pot = np.log(P/self.norm_factor + self.epsilon) / self.beta
        bias_pot = - self.gamma * bias_pot if self.explore else self.gamma_prefac * bias_pot
        pmf = bias_pot/-self.gamma_prefac if not self.explore else bias_pot
        pmf -= pmf.min()
        
        return pmf
    

    def shared_bias(self):
        pass


    def weighted_pmf_history2d(
        self,
        cv_x: np.array,
        cv_y: np.array,
        cv_pot: np.array,
        grid_x: np.array,
        grid_y: np.array,
        hist_res: int = 10,
    ) -> np.array:
        """calculate weighted pmf history

        Args:
            cv_x: trajectory of cv x
            cv_y: trajectory of cv y
            hist_res: resolution of history

        Returns:
            pmf_weight_hist: weighted pmf history
            scattered_time: time points of history
        """

        if self.ncoords != 2:
            raise ValueError(" >>> Error: 2D weighted pmf history can only be calculated for two-dimensional CV spaces!")

        dx = 1.0 # Bin size x dimension
        dy = 1.0 # Bin size y dimension
        dx2 = dx / 2.0
        dy2 = dy / 2.0

        n = int(len(cv_x)/hist_res)
        scattered_time = []
        pmf_weight_hist = []
        divisor = self.sum_weights if not self.explore else self.n

        print("Initialize pmf history calculation.")
        for j in range(hist_res):
            print("Iteration ",j+1," of", hist_res, "started.")
            n_sample = j * n + n
            scattered_time.append(n_sample)
            probability_hist = np.zeros((120,80))
            for i,x in enumerate(grid_x): # Loop over grid so that i are bin centers
                for j,y in enumerate(grid_y):
                    indices_hist = np.where(np.logical_and(np.logical_and((cv_x[0:n_sample] >= x - dx2), (cv_x[0:n_sample] < x + dx2)), np.logical_and((cv_y[0:n_sample] >= y - dy2), (cv_y[0:n_sample] < y + dy2))))
                    probability_hist[i,j] = np.sum(np.exp(self.beta * cv_pot[indices_hist[0]])) /divisor
            probability_hist /= np.array(probability_hist).sum()
            potential_hist = -np.log(probability_hist)/self.beta/kJ_to_kcal
            potential_hist -= potential_hist.min()
            potential_hist = np.where(potential_hist==np.inf, 0, potential_hist)
            pmf_weight_hist.append(potential_hist)
        print("Done")
        
        return pmf_weight_hist, scattered_time
    
    
    def weighted_pmf_history1d(
        self,
        cv_x: np.array,
        cv_pot: np.array,
        grid: np.array,
        hist_res: int = 100,
    ) -> np.array:
        """calculate weighted pmf history

        Args:
            cv_x: trajectory of cv x
            hist_res: resolution of history

        Returns:
            pmf_weight_hist: weighted pmf history
            scattered_time: time points of history
        """

        if self.ncoords != 1:
            raise ValueError(" >>> Error: 1D weighted pmf history can only be calculated for one-dimensional CV spaces!")

        dx = 1.0 # Bin size
        dx2 = dx / 2.0

        n = int(len(cv_x)/hist_res)
        scattered_time = []
        pmf_bin_hist = []
        divisor = self.sum_weights if not self.explore else self.n
        print("Initialize pmf history calculation.")
        for j in range(hist_res):
            n_sample = j * n + n
            print("Iteration ",j+1," of", hist_res, "started.")
            scattered_time.append(n_sample)
            probability_hist = np.zeros(len(grid))
            for i,x in enumerate(grid): # Loop over grid so that i are bin centers
                indices_hist = np.where(np.logical_and((cv_x[0:n_sample] >= x - dx2), (cv_x[0:n_sample] < x + dx2)))
                probability_hist[i] = np.sum(np.exp(self.beta * cv_pot[indices_hist])) / divisor
            probability_hist /= probability_hist.sum()
            potential_hist = (-np.log(probability_hist)/self.beta) /kJ_to_kcal
            potential_hist -= potential_hist.min()
            potential_hist = np.where(potential_hist==np.inf, 0, potential_hist)
            pmf_bin_hist.append(potential_hist)
        print("Done.")

        return pmf_bin_hist, scattered_time
