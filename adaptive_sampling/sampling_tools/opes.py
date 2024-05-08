import sys,os
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *
from .utils import correct_periodicity
import numpy as np

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:
        kernel_std: standard deviation of first kernel
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be 
        approximate_norm: enables approximation of norm factor
        merge_threshold: threshold distance for kde-merging in units of std, "np.inf" disables merging
        recursion_merge: enables recursive merging
        convergence_freq: interval of calculating and writing convergency criteria
        bias_factor: allows setting a default bias factor instead of calculating it from energy barrier
    """
    def __init__(
        self,
        kernel_std: np.array,
        *args,
        energy_barr: float = 20.0,
        update_freq: int = 1000,
        approximate_norm: bool = False,
        merge_threshold: float = 1.0,
        recursion_merge: bool = False,
        convergence_freq: int = 1000,
        bias_factor: float = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Constants
        self.energy_barr = energy_barr * kJ_to_kcal
        self.beta = 1/(self.equil_temp * kB_in_atomic * atomic_to_kJmol * kJ_to_kcal)
        if bias_factor == None:
            self.gamma = self.beta * self.energy_barr
        else:
            self.gamma = bias_factor
        self.gamma_prefac = 1 - 1/self.gamma
        self.temperature = self.equil_temp
        self.epsilon = np.exp(((-1) * self.beta * self.energy_barr)/self.gamma_prefac)

        # Initial values
        self.prob_dist = 1.0
        self.deriv_prob_dist = 0.0
        self.sum_weights = np.power(self.epsilon, self.gamma_prefac)
        self.sum_weights_square = self.sum_weights * self.sum_weights
        self.norm_factor = 1/self.sum_weights
        self.md_state = self.the_md.get_sampling_data()
        self.sigma_0 = self.unit_conversion_cv(np.asarray(kernel_std))[0]

        # Simulation Parameters
        self.update_freq = update_freq
        self.converg_freq = convergence_freq
        self.approx_norm = approximate_norm
        self.merge = False if merge_threshold == np.inf else True
        self.merge_threshold = merge_threshold
        self.recursive = recursion_merge
        s, _ = self.get_cv(**kwargs)

        # Kernels
        self.kernel_height = [1.0]
        self.kernel_center = [s]
        self.kernel_sigma = [self.sigma_0]

        # Output
        self.output_sum_weigths = []
        self.output_sum_weigths_square = []
        self.output_norm_factor = []
        self.output_bias_pot = []


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

        # Call update function to place kernel
        if self.md_state.step%self.update_freq == 0:
            self.update_kde(s_new)

        # Calculate new probability
        val_gaussians = self.get_val_gaussian(s_new)
        prob_dist = np.sum(val_gaussians)

        s_diff = (s_new - np.asarray(self.kernel_center))
        # Correct periodicity of spatial distances
        for i in range(self.ncoords):
            s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])

        deriv_prob_dist = np.sum(-val_gaussians * np.divide(s_diff, np.asarray(self.kernel_sigma)).T, axis=1)
        self.prob_dist = prob_dist
        self.deriv_prob_dist = deriv_prob_dist
        if self.verbose and self.md_state.step%(self.update_freq)==0:
            self.show_kernels()
            print("Probabiliy distribution: ", prob_dist, "and its derivative: ", deriv_prob_dist)
            print("val gaussians =  ", val_gaussians)

        # Calculate Potential
        self.potential = self.calc_pot(prob_dist)

        # Calculate forces
        forces = self.calculate_forces(prob_dist, deriv_prob_dist)
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
        s_diff = (s - np.asarray(self.kernel_center))

        # Correct Periodicity of spatial distances
        for i in range(self.ncoords):
            s_diff[:,i] = correct_periodicity(s_diff[:,i], self.periodicity[i])
            
        # Calculate values of Gaussians at center of kernel currently in loop and sum them
        if self.verbose and self.md_state.step%self.update_freq ==0:
            print("S_diff", s_diff)
            print(np.asarray(self.kernel_sigma))
        val_gaussians = np.asarray(self.kernel_height) * \
            np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))

        return val_gaussians


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
        self.N_old = len(self.kernel_center)
        self.S_old = self.sum_weights

        # calculate probability
        prob_dist = np.sum(self.get_val_gaussian(s_new))
        self.prob_dist = prob_dist

        # Calculate bias potential
        potential = self.calc_pot(prob_dist)

        # Calculate weight coefficients
        weigth_coeff = np.exp(self.beta * potential)
        if self.verbose:
            print("Update function called, now placing new kernel and evaluate its proberties")
            print("Probability distribution: ", prob_dist)
            print("potential is: ", potential)
        self.sum_weights += weigth_coeff
        self.sum_weights_square += weigth_coeff * weigth_coeff

        # Bandwidth rescaling
        self.n_eff = np.square(1+self.sum_weights) / (1+self.sum_weights_square)
        sigma_i =  self.sigma_0 * np.power((self.n_eff * (self.ncoords + 2)/4), -1/(self.ncoords + 4))
        height = weigth_coeff * np.prod(self.sigma_0 / sigma_i)

        # Kernel Density 
        if len(self.kernel_center) > 0 and self.md_state.step > 0:
            self.compression_check(height, s_new, sigma_i)

        # Calculate normalization factor
        self.norm_factor = self.calc_norm_factor(approximate = self.approx_norm)


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
        if self.verbose and self.md_state.step%(self.update_freq*100)==0:
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
        S = self.sum_weights
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
            new_uprob = self.norm_factor * self.N_old * self.S_old + delta_uprob
            norm_factor = new_uprob/N/S
            return norm_factor

        else:
            uprob = 0.0
            # Loop over all kernels with all kernels to get exact uprob
            for s in self.kernel_center:
                sum_gaussians =np.sum(self.get_val_gaussian(s))
                uprob += sum_gaussians
            # Get norm factor from exact uprob and set it
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
        self.kernel_height.append(h_new)
        self.kernel_center.append(s_new)
        self.kernel_sigma.append(std_new)

        kernel_min_ind, min_distance = self.calc_min_dist(s_new)
        
        # Recursive merging if enabled and distances under threshold
        while self.merge and np.all(min_distance < self.merge_threshold):

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
            sigma = self.kernel_sigma
        )


    def restart(
        self, 
        filename: str = "restart_wtmeabf"
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
        self.kernel_height = data["height"]
        self.kernel_center = data["center"]
        self.kernel_sigma = data["sigma"]

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
        """calculate pmf on the fly from compressed kernels
        """
        if self.ncoords == 1:
            P = np.zeros_like(self.grid[0])
            for x in range(len(self.grid[0])):
                s_diff = self.grid[0][x] - np.asarray(self.kernel_center)
                for l in range(self.ncoords):
                    s_diff[:,l] = correct_periodicity(s_diff[:,l], self.periodicity[l])
                val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
                P[x] = np.sum(val_gaussians)

        elif self.ncoords == 2:
            P = np.zeros_like(self.grid)
            for x in range(len(self.grid[0,:])):
                for y in range(len(self.grid[1,:])):
                    s_diff = np.array([self.grid[0,x], self.grid[1,y]]) - np.asarray(self.kernel_center)
                    for l in range(self.ncoords):
                        s_diff[:,l] = correct_periodicity(s_diff[:,l], self.periodicity[l])
                    val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
                    P[x,y] = np.sum(val_gaussians)
        else:
            pmf = 0.0
            return pmf
        
        bias_pot = self.gamma_prefac / self.beta * np.log(P/self.norm_factor + self.epsilon)
        pmf = bias_pot/-self.gamma_prefac
        pmf -= pmf.min()
        
        return pmf


    def shared_bias(self):
        pass



