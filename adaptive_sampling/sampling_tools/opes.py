#from enhanced_sampling import EnhancedSampling
from .utils import gaussian_calc
from .utils import distance_calc
import numpy as np


class OPES():

    def __init__(
        self,
        threshold_kde: float = 1.0,
        kernel_var: np.array = np.ones(2),
        kernel_height: np.array = np.ones(2)
    ):
        self.threshold_kde = threshold_kde
        #lists for characteristics of compressed kernels; new ones added in algorithm or deleted if merged
        #var = variance, std = standard deviation
        self.kernels_h = [1.0]
        self.kernels_var = [kernel_var]
        self.kernels_s = [kernel_height]

    #Calculate distances of new point to all existing kernels and determine the kernel with minimum distance
    def calc_min_dist(self,s_new: np.array):
        distance = []
        print("distance: ", distance)
        for i in range(len(self.kernels_h)):
            distance = distance + [distance_calc(s_new,self.kernels_s[i],self.kernels_var[i])]
            print("distance: ", distance)
        kernel_min_ind = distance.index(min(distance))
        print(kernel_min_ind, distance[kernel_min_ind])
        return kernel_min_ind, distance[kernel_min_ind]
    
    #merge kernel min with new kernel and overwrite old kernel min in lists
    def merge_kernels(self,kernel_min_ind, h_new: float, s_new: np.array, var_new: np.array):
        #print("calculate merged height")
        h_merge = self.kernels_h[kernel_min_ind] + h_new
        #print("calculate merged location")
        s_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * self.kernels_s[kernel_min_ind] + h_new * s_new)
        #print("calculate merged std")
        std_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * (np.square(self.kernels_var[kernel_min_ind]) + np.square(self.kernels_s[kernel_min_ind])) + h_new * (np.square(var_new) + np.square(s_new))) - np.square(s_merge)
        #if len(self.kernels_h) < 2:
        #    print("overwriting instead deleting because there is only one kernel in list")
        self.kernels_h[kernel_min_ind] = h_merge
        self.kernels_s[kernel_min_ind] = s_merge
        self.kernels_var[kernel_min_ind] = np.sqrt(std_merge)
        print("merged kernels and overwrote old one in list")
        #else:
        #    del self.kernels_h[kernel_min_ind]
        #    del self.kernels_var[kernel_min_ind]
        #    del self.kernels_s[kernel_min_ind]
        #    print("old kernel deleted")
        return h_merge, s_merge, np.sqrt(std_merge)

    #add new kernel to lists
    def add_kernel_to_compressed(self,h_new: float, s_new: np.array, var_new: np.array):
        self.kernels_h = self.kernels_h + [h_new]
        self.kernels_s = self.kernels_s + [s_new]
        self.kernels_var = self.kernels_var + [var_new]

    #check whether merging or adding should be fulfilled
    def compression_check(self,h_new: float,s_new: np.array,var_new: np.array, recursive:bool=True):
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