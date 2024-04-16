from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from .utils import gaussian_calc
from .utils import distance_calc
import numpy as np

class OPES(EnhancedSampling):
    """on-the-fly probability enhanced sampling
    
    Args:

    """
    def __init__(
        self,
        *args,
        threshold_kde: float = 1.0,
        kernel_var: np.array = np.ones(1),
        kernel_location: np.array = np.ones(1),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.threshold_kde = threshold_kde
        #lists for characteristics of compressed kernels; new ones added in algorithm or deleted if merged
        #var = variance, std = standard deviation
        self.kernels_var = [kernel_var]
        self.kernels_h = [1.0]
        self.kernels_s = [kernel_location]

    #Calculate distances of new point to all existing kernels and determine the kernel with minimum distance
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
    
    #merge kernel min with new kernel and overwrite old kernel min in lists
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
        #print("merged kernels and overwrote old one in list")
        return h_merge, s_merge, np.sqrt(std_merge)

    #add new kernel to lists
    def add_kernel_to_compressed(self,h_new: float, s_new: np.array, var_new: np.array):
        self.kernels_h = self.kernels_h + [h_new]
        self.kernels_s = self.kernels_s + [s_new]
        self.kernels_var = self.kernels_var + [var_new]

    #check whether merging or adding should be fulfilled
    def compression_check(self, s_new: np.array,var_new: np.array, recursive:bool=True):
        h_new = np.prod(1/(var_new * np.sqrt(2 * np.pi)))
        dist_values = self.calc_min_dist(s_new)
        print("minimal distance is: ", dist_values[1])
        if dist_values[1] < self.threshold_kde:
            print("kernel under threshold distance: ", dist_values[0])
            h_new, s_new, var_new = self.merge_kernels(dist_values[0], h_new, s_new, var_new)
            print("remembered merged kernel")
            if recursive and len(self.kernels_h) > 1:
                print("recursion active and enough kernels in list")
                print(self.show_kernel_lists())
                del self.kernels_h[dist_values[0]]
                del self.kernels_var[dist_values[0]]
                del self.kernels_s[dist_values[0]]
                print("deleted merged kernel from list... recursive compression check in progress")
                self.compression_check(h_new, s_new, var_new)
            else:
                print("break recursion because there is only one kernel in list")
                #self.add_kernel_to_compressed(h_new,s_new,var_new)
        else:
            print("add!")
            self.add_kernel_to_compressed(h_new,s_new,var_new)

    #for testing
    def show_kernel_lists(self):
        print(self.kernels_h)
        print(self.kernels_s)
        print(self.kernels_var)

    

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