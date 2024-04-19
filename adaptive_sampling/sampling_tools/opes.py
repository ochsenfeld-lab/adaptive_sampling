from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from .utils import Kernel
from .utils import gaussian_calc
from .utils import distance_calc
from adaptive_sampling.units import *
import numpy as np

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:
        kernel_var: initial variance of first kernel
        threshold_kde: treshold distance for kde algorithm to trigger merging of kernels
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be placed
        approximate_norm: toggel approximation of norm factor
        merge_kernels: enables merging
        recursion_merge: enables recursive merging
        
    """
    def __init__(
        self,
        kernel_var: np.array,
        *args,
        threshold_kde: float = 3.0,
        energy_barr: float = 20.0,
        update_freq: int = 5000,
        approximate_norm: bool = False,
        merge_kernels: bool = False,
        recursion_merge: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta = 1/(self.equil_temp * kB_in_atomic * atomic_to_kJmol * kJ_to_kcal)
        self.threshold_kde = threshold_kde
        self.norm_factor = 1.0
        self.exact_norm_factor = 0.0
        self.gamma = self.beta * energy_barr
        self.gamma_prefac = 1 - 1/self.gamma
        self.temperature = self.equil_temp
        self.epsilon = np.exp(((-1) * self.beta * energy_barr)/self.gamma_prefac)
        self.prob_dist = 1.0
        self.sum_weights = 1.0
        self.sum_weights_square = 1.0
        s, _ = self.get_cv(**kwargs)
        self.kernel_list = [] #[Kernel(1.0, s, kernel_var)]
        self.sigma_0 = kernel_var
        self.update_freq = update_freq
        self.n_step = 0
        self.in_recursion = False
        self.approx_norm = approximate_norm
        self.merge = merge_kernels
        self.recursive = recursion_merge
        self.md_step = 0



    def calc_min_dist(self,s_new: np.array):
        """calculate all distance between sample point and deployed gaussians and find the minimal

        Args:
            s_new: sampling point

        Returns:
            kernel_min_ind: index of nearest gaussian with respect to initialized lists
            distance[kernel_min_ind]: distance to nearest gaussian as float
        """
        distance = []
        for k in range(len(self.kernel_list)-1):
            distance += [distance_calc(s_new,self.kernel_list[k].center, self.kernel_list[k].sigma, self.periodicity)]
        kernel_min_ind = distance.index(min(distance))
        #print("kernel with minimal distance is ", kernel_min_ind+1, "in ", distance[kernel_min_ind])
        return kernel_min_ind, distance[kernel_min_ind]
    
    def merge_kernels(self, kernel_min_ind: int, h_new: float, s_new: np.array, var_new: np.array) -> list:
        """merge two kernels calculating the characteristics of a new one with respect to the old ones and overwrite old one with merged

        Args:
            kernel_mind_ind: index of nearest kernel, which is to be merged
            h_new: height of new kernel
            s_new: center of new kernel
            var_new: variance of new kernel

        Returns:
            height, center and variance of merged kernel       
        """
        h_merge = self.kernel_list[kernel_min_ind].height + h_new
        s_merge = (1.0/h_merge)*(self.kernel_list[kernel_min_ind].height * self.kernel_list[kernel_min_ind].center + h_new * s_new)
        std_merge = (1.0/h_merge)*(self.kernel_list[kernel_min_ind].height * (np.square(self.kernel_list[kernel_min_ind].sigma) + np.square(self.kernel_list[kernel_min_ind].center)) + h_new * (np.square(var_new) + np.square(s_new))) - np.square(s_merge)
        self.kernel_list.append(Kernel(h_merge, s_merge, np.sqrt(std_merge)))
        del self.kernel_list[kernel_min_ind]
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("merge successful: ", [k.height for k in self.kernel_list], [k.center for k in self.kernel_list])
        return h_merge, s_merge, np.sqrt(std_merge)

    def add_kernel_to_compressed(self,h_new: float, s_new: np.array, var_new: np.array):
        """generate and add new kernel object on sampling point to list of compressed ones

        Args:
            h_new: height of new kernel
            s_new: center of new kernel
            var_new: variance of new kernel
        """
        self.kernel_list.append(Kernel(h_new, s_new, var_new))

    def compression_check(self, heigth: float, s_new: np.array, var_new: np.array):
        """kde compression check: new kernel added in update function is tested for being to near to an existing compressed one;
         if so, merge kernels and delete added, check recursive, otherwise skip

        Args:
            heigth: weighted heigth of new kernel
            s_new: center of new kernel
            var_new: variance of new kernel
            recursive: boolean, recursion needed, default True
        """
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("kernel to check: ", heigth, s_new)
            print("in recursion: ", self.in_recursion)
        h_new = heigth
        if not self.merge or self.md_step == 0:
            return
        threshold = self.threshold_kde * np.square(var_new)
        self.show_kernels()
        dist_values = self.calc_min_dist(s_new)
        #print("minimal distance is: ", dist_values[1])
        if np.all(dist_values[1] < threshold):
            if self.verbose and self.md_step%(self.update_freq*100)==0:
                print("kernel under threshold distance ", threshold, "in distance: ", dist_values[1])
            h_new, s_new, var_new = self.merge_kernels(dist_values[0], h_new, s_new, var_new)
            index_merged_kernel = dist_values[0]
            l = len(self.kernel_list)
            self.show_kernels()
            #print("deleted kernel to merge")
            self.show_kernels()
            #print("remembered merged kernel")
            if self.recursive and len(self.kernel_list) > 1:
                self.in_recursion = True
                if self.verbose and self.md_step%(self.update_freq*100)==0:
                    print("recursion active and enough kernels in list")
                    print("recursive compression check in progress")
                self.compression_check(h_new, s_new, var_new)
            else:
                if self.verbose and self.md_step%(self.update_freq*100)==0:
                    print("break recursion because there is only one kernel in list")
                #self.add_kernel_to_compressed(h_new, s_new, var_new)
        else:
            if self.verbose and self.md_step%(self.update_freq*100)==0:
                print("kernel over threshold distance ", threshold, "in distance: ", dist_values[1])
                if self.in_recursion == True:
                    print("nothing to merge recursive")
            self.in_recursion = False
            #print("kernel stays in list, no merging and deleting required")


    def calc_probab_distr(self, s_prob: np.array, index_modif: int = 0, require_grad: bool = True) -> list:
        """on the fly calculation of not normalized probability distribution

        Args:
            s_prob: location in which probability distribution shall be evaluated
            index_modif: possibility to variate the upper sum boundary
            require_grad: decides, whether derivative of probability distribution is saved

        Returns:
            prob: probability distribution for location s_prob
            deriv_prob: derivative of probability distribtuion in s_prob
        """
        prob = 0.0
        deriv_prob = 0.0
        for k in range(len(self.kernel_list) - index_modif):
            prob += self.kernel_list[k].evaluate(s_prob)[0]
            deriv_prob += self.kernel_list[k].evaluate(s_prob)[1]
        #print("probability distribution is: ", deriv_prob)
        return prob, deriv_prob


    def calc_pot(self, prob_dist: float, index_modif: int = 0):
        """calculate the potential of given location
        
        Args:
            prob_dist: local probability distribution on which potential shall be calculated

        Returns:
            potential: potential for given probability distribution
        """
        potential = (self.gamma_prefac / self.beta)*np.log(((1/self.norm_factor)*(prob_dist))+ self.epsilon)
        return potential
    
    def calculate_forces(self, prob_dist:float, deriv_prob_dist: float):
        """calculate the forces as derivative of potential

        Args:
            prob_dist: probability disitrbution on wanted location and its derivative 

        Returns:
            deriv_pot: derivative of potential for location s
        """
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("Calculate forces function called")
            print("beta: ",self.beta,"gamma prefactor: ",self.gamma_prefac,"probability dist: ",prob_dist,"norm factor: ",self.norm_factor,"epsilon: ", self.epsilon)
        deriv_pot = (self.gamma_prefac) * (1/self.beta) * (1/ ((prob_dist / self.norm_factor) + self.epsilon))
        deriv_pot *= (deriv_prob_dist / self.norm_factor)
        return deriv_pot

    def calc_norm_factor(self, index_modif: int = 0):
        """calculate the norm factor with respect to existing gaussians
        """
        S = self.sum_weights
        N = len(self.kernel_list)
        sum_gaussians = 0.0
        for k in range(len(self.kernel_list) - index_modif):
            sum_gaussians += self.kernel_list[k].evaluate(self.kernel_list[k].center)[0]
        delta_norm_factor = (1/(S * N)) * sum_gaussians
        return self.norm_factor + delta_norm_factor

    def update_kde(self, s_new: np.array):
        """main function in algorithm; calls compression and calculates weigths
        
        Args:
            s_new: center of new gaussian

        """
        prob_dist = self.calc_probab_distr(s_new)
        weigth_coeff = np.exp(self.beta * prob_dist[0])
        if self.verbose:
            print("Update function called, now placing new kernel and evaluate its proberties")
            print("Probability distribution: ", prob_dist)
            print("weigth coefficient is: ", weigth_coeff)
        self.sum_weights += weigth_coeff
        self.sum_weights_square += weigth_coeff * weigth_coeff
        self.n_eff = np.square(self.sum_weights) / self.sum_weights_square
        sigma_i =  self.sigma_0 * np.power((self.n_eff * (self.ncoords + 2)/4), -1/(self.ncoords + 4))
        height = weigth_coeff #* np.prod(self.sigma_0 / sigma_i)
        self.add_kernel_to_compressed(height, s_new, sigma_i)
        if self.approx_norm:
            self.norm_factor = self.calc_norm_factor()
        else:
            self.norm_factor = self.calculate_exact_norm_factor()
        self.compression_check(height, s_new, sigma_i)

    def show_kernels(self):
        """for testing print kernels
        """
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("kernels: ", [k.height for k in self.kernel_list],[k.center for k in self.kernel_list],[k.sigma for k in self.kernel_list])
        else:
            pass

    def step_bias(self, **kwargs):
        """pulls sampling data and CV, drops a gaussian every Udate_freq MD-steps, calculates the bias force

        Args:
            s_new: sampling point

        Returns:
            bias_force: bias_force on location in CV space in atomic units
        """
        md_state = self.the_md.get_sampling_data()
        (s_new, delta_s_new) = self.get_cv(**kwargs)
        self.md_step = md_state.step
        if md_state.step%self.update_freq == 0:
            #if self.verbose and self.md_step%(self.update_freq*100)==0:
            #    print("step_bias adds new kernel with", s_new)
            self.update_kde(s_new)
        prob_dist, deriv_prob_dist = self.calc_probab_distr(s_new)
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            self.show_kernels()
            print("Probabiliy distribution: ", prob_dist, "and its derivative: ", deriv_prob_dist)
        self.potential = self.calc_pot(prob_dist)
        forces = self.calculate_forces(prob_dist, deriv_prob_dist)
        bias_force = np.zeros_like(md_state.coords, dtype=float)
        for i in range(self.ncoords):
            bias_force += forces[i] * delta_s_new[i]
        bias_force = bias_force / kJ_to_kcal / atomic_to_kJmol
        #print("the bias force is: ",  bias_force)
        return bias_force
    
    def calculate_exact_norm_factor(self):
        S = self.sum_weights
        N = len(self.kernel_list)
        sum_gaussians = 0.0
        for k in range(len(self.kernel_list)):
            for k_s in range(len(self.kernel_list)):
                #print("value of Gaussian: ", self.kernel_list[k].evaluate(self.kernel_list[k_s].center)[0])
                sum_gaussians += self.kernel_list[k].evaluate(self.kernel_list[k_s].center)[0]
                #print(sum_gaussians)
        if self.verbose and self.md_step%(self.update_freq*100)==0:
            print("Calculate exact norm factor function called")
            self.show_kernels()
            print("double sum over the gaussians gives: ",sum_gaussians)
        delta_norm_factor = (1/(S * N)) * sum_gaussians
        return delta_norm_factor
        
    
    def get_pmf(self):
        P = np.zeros_like(self.grid[0])
        for i in range(len(self.grid[0])):
            for k in self.kernel_list:
                P[i] += k.evaluate(self.grid[0][i])[0]
        pmf = self.gamma_prefac / self.beta * np.log(P/self.norm_factor + self.epsilon)

    def shared_bias(self):
        pass

    def write_restart(self):
        pass

    def restart(self):
        pass

    def write_traj(self):
        pass