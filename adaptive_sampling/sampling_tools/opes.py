from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from .utils import Kernel
from .utils import distance_calc
from adaptive_sampling.units import *
from .utils import correct_periodicity
import numpy as np

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:
        kernel_var: initial variance of first kernel
        threshold_kde: treshold distance for kde algorithm to trigger merging of kernels
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be 
        approximate_norm: toggel approximation of norm factor
        merge_kernels: enables merging
        recursion_merge: enables recursive merging
        
    """
    def __init__(
        self,
        kernel_std: np.array,
        *args,
        threshold_kde: float = 1.0,
        energy_barr: float = 20.0,
        update_freq: int = 5000,
        approximate_norm: bool = False,
        merge_threshold: bool = np.inf,
        recursion_merge: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.energy_barr = energy_barr * kJ_to_kcal
        self.beta = 1/(self.equil_temp * kB_in_atomic * atomic_to_kJmol * kJ_to_kcal)
        self.threshold_kde = threshold_kde
        self.norm_factor = 1.0
        self.exact_norm_factor = 0.0
        self.gamma = self.beta * self.energy_barr
        self.gamma_prefac = 1 - 1/self.gamma
        self.temperature = self.equil_temp
        self.epsilon = np.exp(((-1) * self.beta * self.energy_barr)/self.gamma_prefac)
        self.prob_dist = 1.0
        self.deriv_prob_dist = 0.0
        self.sum_weights = np.power(self.epsilon, self.gamma_prefac)
        self.sum_weights_square = self.sum_weights * self.sum_weights
        self.output_sum_weigths = []
        self.output_sum_weigths_square = []
        self.output_norm_factor = []
        self.output_bias_pot = []
        s, _ = self.get_cv(**kwargs)
        #self.kernel_list = []
        self.kernel_height = []
        self.kernel_center = []
        self.kernel_sigma = []
        self.sigma_0 = kernel_std
        self.update_freq = update_freq
        self.n_step = 0
        self.approx_norm = approximate_norm
        self.merge = False if merge_threshold == np.inf else True
        self.merge_threshold = merge_threshold
        self.recursive = recursion_merge
        self.md_step = 0



    def calc_min_dist(self,s_new: np.array) -> tuple:
        """calculate all distance between sample point and deployed gaussians and find the minimal

        Args:
            s_new: current cv point

        Returns:
            kernel_min_ind: index of nearest gaussian with respect to initialized lists
            distance[kernel_min_ind]: distance to nearest gaussian as float
        """
        if not hasattr(s_new, "__len__"):
            raise ValueError("Wrong Input: Not array!")
        s_diff = np.absolute(s_new - np.asarray(self.kernel_center))
        for i,p in enumerate(self.periodicity):
            s_diff[i] = correct_periodicity(s_diff[i], p)
        distance = np.sqrt(np.sum(np.square(s_diff/self.kernel_sigma)))
        kernel_min_ind = distance.index(min(distance))
        #print("kernel with minimal distance is ", kernel_min_ind+1, "in ", distance[kernel_min_ind])
        return kernel_min_ind, distance[kernel_min_ind]
    
    def merge_kernels(self, kernel_min_ind: int, h_new: float, s_new: np.array, std_new: np.array) -> list:
        """merge two kernels calculating the characteristics of a new one with respect to the old ones and overwrite old one with merged

        Args:
            kernel_mind_ind: index of nearest kernel, which is to be merged
            h_new: height of new kernel
            s_new: center of new kernel
            std_new: standard deviation of new kernel

        Returns:
            height, center and standard deviation of merged kernel       
        """
        del self.kernel_height[-1]
        del self.kernel_center[-1]
        del self.kernel_sigma[-1]
        #print(kernel_min_ind)
        #print(self.kernel_height)
        h_merge = self.kernel_height[kernel_min_ind] + h_new
        s_merge = (1.0/h_merge)*(self.kernel_height[kernel_min_ind] * self.kernel_center[kernel_min_ind] + h_new * s_new)
        var_merge = (1.0/h_merge)*(self.kernel_height[kernel_min_ind] * (np.square(self.kernel_sigma[kernel_min_ind]) + np.square(self.kernel_center[kernel_min_ind])) + h_new * (np.square(std_new) + np.square(s_new))) - np.square(s_merge)
        self.kernel_height.append(h_merge)
        self.kernel_center.append(s_merge)
        self.kernel_sigma.append(np.sqrt(var_merge))
        del self.kernel_list[kernel_min_ind]
        if self.verbose:
            print("merge successful: ")
            self.show_kernels()
        return h_merge, s_merge, np.sqrt(var_merge)


    def compression_check(self, height: float, s_new: np.array, std_new: np.array):
        """kde compression check: new kernel added in update function is tested for being to near to an existing compressed one;
         if so, merge kernels and delete added, check recursive, otherwise skip

        Args:
            heigth: weighted heigth of new kernel
            s_new: center of new kernel
            std_new: stndard deviation of new kernel
            recursive: boolean, recursion needed, default True
        """
        print("Kernel Center Test!!:",np.asarray(self.kernel_center)[0:-1])
        #if len(self.kernel_center)>0:
        print(s_new)
        #s_diff = np.absolute(s_new - np.asarray(self.kernel_center)[0:-1])
        #else:
        s_diff = np.absolute(s_new - np.asarray(self.kernel_center))
        #if len(s_diff)>0:
        for i,p in enumerate(self.periodicity[0:-1]):
            s_diff[i] = correct_periodicity(s_diff[i], p)
        distance = np.sqrt(np.sum(np.square(np.divide(s_diff, self.kernel_sigma)), axis=1))
        kernel_min_ind = np.argmin(distance)
        min_distance = distance[kernel_min_ind]
        if self.verbose:
            print("calculated distances are: ", distance)
            #print("kernel with minimal distance is ", kernel_min_ind+1, "in ", distance[kernel_min_ind])
            #print(kernel_min_ind, min_distance)
        if self.verbose:
            self.show_kernels()
            print("kernel to check: ", height, s_new)
        h_new = height
        if not self.merge or self.md_step == 0:
            if self.verbose:
                print("not merging or md_step = 0")
            return
        if np.all(min_distance < self.merge_threshold):
            if self.verbose:
                print("kernel under threshold distance ", self.merge_threshold, "in distance: ", min_distance)
            h_new, s_new, std_new = self.merge_kernels(kernel_min_ind, h_new, s_new, std_new)
            if self.recursive and len(self.kernel_list) > 1:
                if self.verbose:
                    print("recursion active and enough kernels in list")
                    print("recursive compression check in progress")
                self.compression_check(h_new, s_new, std_new)
            else:
                if self.verbose:
                    if self.merge:
                        print("break recursion because there is only one kernel in list")
                    else:
                        print("not recursive merging")
        else:
            if self.verbose:# and self.md_step%(self.update_freq*100)==0:
                print("kernel over threshold distance ", threshold, "in distance: ", dist_values[1])


    def calc_probab_distr(self, s_prob: np.array, require_grad: bool = True) -> list:
        """on the fly calculation of not normalized probability distribution

        Args:
            s_prob: location in which probability distribution shall be evaluated
            require_grad: decides, whether derivative of probability distribution is saved

        Returns:
            prob: probability distribution for location s_prob
            deriv_prob: derivative of probability distribtuion in s_prob
        """
        prob = 1.0
        deriv_prob = 0.0
        for k in range(len(self.kernel_list)):
            prob += self.kernel_list[k].evaluate(s_prob)[0]
            deriv_prob += self.kernel_list[k].evaluate(s_prob)[1]
        #print("probability distribution is: ", deriv_prob)
        return prob, deriv_prob


    def calc_pot(self, prob_dist: float):
        """calculate the potential of given location
        
        Args:
            prob_dist: local probability distribution on which potential shall be calculated

        Returns:
            potential: potential for given probability distribution
        """
        potential = (self.gamma_prefac / self.beta) * np.log(prob_dist / self.norm_factor + self.epsilon)
        return potential
    
    def calculate_forces(self, prob_dist:float, deriv_prob_dist: float):
        """calculate the forces as derivative of potential

        Args:
            prob_dist: probability disitrbution on wanted location and its derivative 

        Returns:
            deriv_pot: derivative of potential for location s
        """
        if self.verbose:
            print("Calculate forces function called")
            print("prob_dist: ", prob_dist, deriv_prob_dist)
            print("norm factor: ", self.norm_factor)
            print("beta: ",self.beta,"gamma prefactor: ",self.gamma_prefac,"probability dist: ",prob_dist,"norm factor: ",self.norm_factor,"epsilon: ", self.epsilon)
        deriv_pot = (self.gamma_prefac/self.beta) * (1/ ((prob_dist / self.norm_factor) + self.epsilon)) * (deriv_prob_dist / self.norm_factor)
        if self.verbose:
            print("derivate of pot: ", deriv_pot)
        return deriv_pot[0]

    def update_kde(self, s_new: np.array):
        """main function in algorithm; calls compression and calculates weigths
        
        Args:
            s_new: center of new gaussian

        """
        # Calculate probability
        if len(self.kernel_center) > 0:
            s_diff = np.absolute(s_new - np.asarray(self.kernel_center))
            for i,p in enumerate(self.periodicity):
                s_diff[i] = correct_periodicity(s_diff[i], p)
            val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
            prob_dist = np.sum(val_gaussians)
            self.prob_dist = prob_dist
        else:
            prob_dist = self.prob_dist

        # Calculate bias potential
        potential = self.calc_pot(prob_dist)

        # Calculate weight coefficients
        weigth_coeff = np.exp(self.beta * potential)
        if self.verbose:
            print("Update function called, now placing new kernel and evaluate its proberties")
            print("Probability distribution: ", prob_dist)
            print("potential is: ", potential)
            print("weigth coefficient is: ", weigth_coeff)
        self.sum_weights += weigth_coeff
        self.sum_weights_square += weigth_coeff * weigth_coeff

        # Bandwidth rescaling
        self.n_eff = np.square(self.sum_weights) / self.sum_weights_square
        sigma_i =  self.sigma_0 * np.power((self.n_eff * (self.ncoords + 2)/4), -1/(self.ncoords + 4))
        height = weigth_coeff * np.prod(self.sigma_0 / sigma_i)

        # Add new Kernel at current cv point
        self.kernel_height.append(height)
        self.kernel_center.append(s_new)
        self.kernel_sigma.append(sigma_i)

        # Calculate normalization factor
        s_diff = np.absolute(s_new - np.asarray(self.kernel_center))
        for i,p in enumerate(self.periodicity):
            s_diff[i] = correct_periodicity(s_diff[i], p)
        val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))

        if self.approx_norm:
            self.norm_factor = self.calc_norm_factor(s_new, val_gaussians)
        else:
            self.norm_factor = self.calculate_exact_norm_factor()

        # Kernel Density Compression
        self.compression_check(height, s_new, sigma_i)

    def show_kernels(self):
        """for testing print kernels
        """
        if self.verbose: #and self.md_step%(self.update_freq*100)==0:
            print("Kernels: ")
            print("heights ", self.kernel_height)
            print("centers ", self.kernel_center)
            print("sigmas ", self.kernel_sigma)
        else:
            pass

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
            s_new: sampling point

        Returns:
            bias_force: bias_force on location in CV space in atomic units
        """

        # Load md data
        md_state = self.the_md.get_sampling_data()
        (s_new, delta_s_new) = self.get_cv(**kwargs)
        self.md_step = md_state.step


        # Call update function to place kernel
        if md_state.step%self.update_freq == 0:
            if self.verbose:
                print("sum weights is: ", self.sum_weights)
                print("norm_factor is:" , self.norm_factor)
                print("gamma prefactor, epsilon and beta: ", self.gamma_prefac, self.epsilon, self.beta)
                self.show_kernels()
            self.update_kde(s_new)

        # Calculate new probability
        s_diff = np.absolute(s_new - np.asarray(self.kernel_center).T)
        for i,p in enumerate(self.periodicity):
            s_diff[i] = correct_periodicity(s_diff[i], p)
        val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
        prob_dist = np.sum(val_gaussians)
        deriv_prob_dist = -np.asarray(self.kernel_height) * np.exp(-0.5 * np.square(np.divide(s_diff, np.asarray(self.kernel_sigma)))) * np.divide(s_diff, np.asarray(self.kernel_sigma))
        self.prob_dist = prob_dist
        self.deriv_prob_dist = deriv_prob_dist
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            self.show_kernels()
            print("Probabiliy distribution: ", prob_dist, "and its derivative: ", deriv_prob_dist)

        # Calculate Potential
        self.potential = self.calc_pot(prob_dist)

        # Calculate forces
        forces = self.calculate_forces(prob_dist, deriv_prob_dist)
        bias_force = np.zeros_like(self.the_md.coords, dtype=float)
        for i in range(self.ncoords):
            bias_force += forces[i] * delta_s_new[i]
        bias_force = bias_force / kJ_to_kcal / atomic_to_kJmol

        # Save values for traj
        self.traj = np.append(self.traj, [s_new], axis=0)
        self.epot.append(md_state.epot)
        self.temp.append(md_state.temp)
        self.output_bias_pot.append(self.potential)
        self.output_sum_weigths.append(self.sum_weights)
        self.output_sum_weigths_square.append(self.sum_weights_square)
        self.output_norm_factor.append(self.norm_factor)


        # Write output
        if md_state.step % self.out_freq == 0:

            if write_traj:
                self.write_traj(filename=traj_file)

            if write_output:
                #self.get_pmf()
                #self.write_output(filename=output_file)
                self.write_restart(filename=restart_file)

        return bias_force
    
    
    def calc_norm_factor(self, s_new: np.array, val_gaussians: np.array):
        """calculate the norm factor with respect to existing gaussians
        """
        S = self.sum_weights
        N = len(self.kernel_center)
        sum_gaussians = np.sum(val_gaussians)
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("Calculate exact norm factor function called")
            self.show_kernels()
            print("sum over the gaussians gives: ",sum_gaussians)
        delta_norm_factor = (1/(S * N)) * sum_gaussians
        return self.norm_factor + delta_norm_factor

    def calculate_exact_norm_factor(self):
        S = self.sum_weights
        N = len(self.kernel_center)
        sum_sum_gaussians = 0.0
        for s in self.kernel_center:
            s_diff = np.absolute(s - np.asarray(self.kernel_center))
            for i,p in enumerate(self.periodicity):
                s_diff[i] = correct_periodicity(s_diff[i], p)
            val_gaussians = np.asarray(self.kernel_height) * np.exp(-0.5 * np.sum(np.square(np.divide(s_diff, np.asarray(self.kernel_sigma))),axis=1))
            sum_gaussians =np.sum(val_gaussians)
            sum_sum_gaussians += sum_gaussians
        
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("Calculate exact norm factor function called")
            self.show_kernels()
            print("double sum over the gaussians gives: ",sum_sum_gaussians)
        norm_factor = (1/(S * N)) * sum_sum_gaussians
        return norm_factor
        
    
    def get_pmf(self):
        pass

    def shared_bias(self):
        pass

    def write_restart(self, filename: str="restart_opes"):
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

    def restart(self, filename: str = "restart_wtmeabf", restart_ext_sys: bool=False):
        """restart from restart file

        Args:
            filename: name of restart file
            restart_ext_sys: restart coordinates and momenta of extended system
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.sum_weights = float(data["sum_weigths"])
        self.sum_weights_square = float(data["sum_weigths_square"])
        self.norm_factor = float(data["norm_factor"])
        self.kernel_list = [Kernel(heigth, center, sigma) for heigth, center, sigma in zip(data['heigth'], data['center'], data['sigma'])]


        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def write_traj(self, filename: str = 'CV_traj.dat'):
        """save trajectory for post-processing"""

        data = {}
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp
        data["Bias pot"] = self.output_bias_pot

        self._write_traj(data, filename=filename)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.output_bias_pot = [self.output_bias_pot[-1]]


