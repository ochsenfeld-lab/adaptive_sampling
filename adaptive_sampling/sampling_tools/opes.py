from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from .utils import gaussian_calc
from .utils import distance_calc
import numpy as np

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:
        threshold_kde: treshold distance for kde algorithm to trigger merging of kernels
        kernel_var: list of variances of kernels, index global for class
        kernel_location: list of centers of compressed kernels
        kernel_weigth_coeff: list of weigthing coefficients used in algorithm
        norm_factor = current normalization factor
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
            [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinment of CVs to the range of interest with harmonic walls
        output_freq: frequency in steps for writing outputs
        multiple_walker: share bias with other simulations via buffer file
        periodicity: periodicity of CVs, [[lower_boundary0, upper_boundary0], ...]

    """
    def __init__(
        self,
        *args,
        threshold_kde: float = 3.0,
        kernel_var: np.array = np.ones(2),
        kernel_location: np.array = np.ones(2),
        kernel_weigth_coeff: float = 1.0,
        norm_factor: float = 1.0,
        gamma: float = 0.5,
        temperature: float = 300,
        beta: float = 1.0,
        epsilon:float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.threshold_kde = threshold_kde
        self.kernels_var = [kernel_var]
        self.kernels_h = [1.0]
        self.kernels_s = [kernel_location]
        self.kernels_weigth_coeff = [kernel_weigth_coeff]
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.temperature = temperature
        self.beta = 1/temperature
        self.epsilon = epsilon

    def calc_min_dist(self,s_new: np.array):
        """calculate all distance between sample point and deployed gaussians and find the minimal

        Args:
            s_new: sampling point

        Returns:
            kernel_min_ind: index of nearest gaussian with respect to initialized lists
            distance[kernel_min_ind]: distance to nearest gaussian as float
        """
        distance = []
        print("distance: ", distance)
        for i in range(len(self.kernels_h)):
            distance = distance + [distance_calc(s_new,self.kernels_s[i],self.kernels_var[i],self.periodicity)]
            print("distance: ", distance)
        kernel_min_ind = distance.index(min(distance))
        print(kernel_min_ind, distance[kernel_min_ind])
        return kernel_min_ind, distance[kernel_min_ind]
    
    def merge_kernels(self,kernel_min_ind, h_new: float, s_new: np.array, var_new: np.array):
        """merge two kernels calculating the characteristics of a new one with respect to the old ones and overwrite old one with merged

        Args:
            kernel_mind_ind: index of nearest kernel, which is to be merged
            h_new: height of new kernel
            s_new: center of new kernel
            var_new: variance of new kernel

        Returns:
            height, center and variance of merged kernel as list        
        """
        h_merge = self.kernels_h[kernel_min_ind] + h_new
        s_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * self.kernels_s[kernel_min_ind] + h_new * s_new)
        std_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * (np.square(self.kernels_var[kernel_min_ind]) + np.square(self.kernels_s[kernel_min_ind])) + h_new * (np.square(var_new) + np.square(s_new))) - np.square(s_merge)
        self.kernels_h[kernel_min_ind] = h_merge
        self.kernels_s[kernel_min_ind] = s_merge
        self.kernels_var[kernel_min_ind] = np.sqrt(std_merge)
        return h_merge, s_merge, np.sqrt(std_merge)

    def add_kernel_to_compressed(self,h_new: float, s_new: np.array, var_new: np.array):
        """add new kernel on sampling point to compressed ones

        Args:
            h_new: height of new kernel
            s_new: center of new kernel
            var_new: variance of new kernel

        Returns:
            nothing, adds kernels to lists
        """
        self.kernels_s = self.kernels_s + [s_new]
        self.kernels_var = self.kernels_var + [var_new]
        self.kernels_h = self.kernels_h + [h_new]

    def compression_check(self, s_new: np.array,var_new: np.array, recursive:bool=True):
        """kde algorithm: check whether merging or adding should be fulfilled acording to threshold distance 
        and also check if after compression there would be enough kernels in list; if adding kernel, update normalization

        Args:
            s_new: center of new kernel
            var_new: variance of new kernel
            recursive: boolean, recursion needed, default True

        Returns:
            nothing, deletes old kernels in lists
        """
        h_new = np.prod(1/(var_new * np.sqrt(2 * np.pi)))
        threshold = self.threshold_kde * var_new
        dist_values = self.calc_min_dist(s_new)
        #print("minimal distance is: ", dist_values[1])
        if np.all(dist_values[1] < threshold):
            #print("kernel under threshold distance: ", dist_values[0])
            h_new, s_new, var_new = self.merge_kernels(dist_values[0], h_new, s_new, var_new)
            #print("remembered merged kernel")
            if recursive and len(self.kernels_h) > 1:
                #print("recursion active and enough kernels in list")
                #print(self.show_kernel_lists())
                del self.kernels_h[dist_values[0]]
                del self.kernels_var[dist_values[0]]
                del self.kernels_s[dist_values[0]]
                #print("deleted merged kernel from list... recursive compression check in progress")
                self.compression_check(h_new, s_new, var_new)
            else:
                print("break recursion because there is only one kernel in list")
                #self.add_kernel_to_compressed(h_new,s_new,var_new)
        else:
            print("add!")
            self.add_kernel_to_compressed(h_new,s_new,var_new)

    def show_kernel_lists(self):
        """for testing: print lists of kernel characteristics

        Arg:

        Returns: 
            lists of kernels
        """
        print(self.kernels_h)
        print(self.kernels_s)
        print(self.kernels_var)

    def calc_prob_dist(self, s_prob_dist: np.array, require_grad: bool = True):
        """on the fly calculation of not normalized probability distribution

        Args:
            s_prob_dist: location in which probability distribution shall be evaluated

        Returns:
            prob_dist: probability distribution for location s
        """
        prob_dist = 1.0
        nominator = 0
        nom_grad = np.zeros(self.ncoords)
        self.sum_weights= 0
        self.sum_weights_quad = 0
        for k in range(len(self.kernels_weigth_coeff)):
            if k == 0:
                weight = 1.0
            else:
                weight = np.exp(self.beta * self.calculate_potential(prob_dist))
            if require_grad == False:
                nominator += weight * gaussian_calc(self.kernels_s[k], self.kernels_var[k], s_prob_dist, periodicity=[None], requires_grad=None)[0]
            else: 
                pot, grad_pot = gaussian_calc(self.kernels_s[k], self.kernels_var[k], s_prob_dist, periodicity=[None], requires_grad=True)
                nominator += weight * pot
                nom_grad += weight * grad_pot
            print("gaussian: ", gaussian_calc(self.kernels_s[k], self.kernels_var[k], s_prob_dist, periodicity=[None], requires_grad=None)[0])
            print("counter: ", nominator)
            self.sum_weights += weight
            print("sum weights: ", self.sum_weights)
            self.sum_weights_quad += weight * weight
            print("denominator: ", weight)
            prob_dist = nominator/self.sum_weights
        self.deriv_prob_dist = nom_grad / self.sum_weights
        print("probability distribution is: ", prob_dist)
        return prob_dist

    def calculate_potential(self, prob_dist: float):
        """calculate the potential of given location
        
        Args:
            s_pot: location, in which potential is wanted

        Returns:
            potential: potential in location s
        """
        potential = (1-(1/self.gamma))*(1/self.beta)*np.log(((1/self.norm_factor)*(prob_dist))+ self.epsilon)
        print("potential is: ",potential)
        return(potential)
    
    def calculate_forces(self, prob_dist:float):
        deriv_pot = (1-(1/self.gamma)) * (1/self.beta) * (1/ ((prob_dist / self.norm_factor) + self.epsilon))
        deriv_pot *= (self.deriv_prob_dist / self.norm_factor)
        return deriv_pot

    def calc_norm_factor(self, s_new: np.array):
        S = self.sum_weights
        N = len(self.kernels_s)
        sum_gaussians = 0.0
        for k in range(len(self.kernels_s)):
            sum_gaussians += gaussian_calc(self.kernels_s[k],self.kernels_var[k],s_new,periodicity=[None])[0]
        delta_norm_factor = (1/(S * N)) * sum_gaussians
        self.norm_factor += delta_norm_factor

            
    def add_kernel():
        pass
    
    def step_bias(self):
        pass

    def get_pmf(self):
        pass

    def shared_bias(self):
        pass

    def write_restart(self):
        pass

    def restart(self):
        pass

    def write_traj(self):
        pass