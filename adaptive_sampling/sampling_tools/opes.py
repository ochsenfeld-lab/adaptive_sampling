#from enhanced_sampling import EnhancedSampling
from .utils import gaussian_calc
from .utils import distance_calc
import numpy as np


class OPES():

    def __init__(
        self,
        threshold_kde = 1
    ):
        self.threshold_kde = threshold_kde
        #lists for characteristics of compressed kernels; new ones added in algorithm or deleted if merged
        #var = variance, std = standard deviation
        self.kernels_h = [1]
        self.kernels_var = [1]
        self.kernels_s = [1]

    #Calculate distances of new point to all existing kernels and determine the kernel with minimum distance
    def calc_min_dist(self,s_new):
        distance = []
        #print(distance)
        for i in range(len(self.kernels_h)):
            distance = distance + [distance_calc(s_new,self.kernels_s[i],self.kernels_var[i])]
            #print(distance)
        kernel_min_ind = distance.index(min(distance))
        #print(kernel_min_ind, distance[kernel_min_ind])
        return kernel_min_ind, distance[kernel_min_ind]
    
    #merge kernel min with new kernel and overwrite old kernel min in lists
    def merge_kernels(self,kernel_min_ind, h_new, s_new, var_new):
        print("calculate merged height")
        h_merge = self.kernels_h[kernel_min_ind] + h_new
        #print(self.kernels_h)        
        #print(self.kernels_s)
        #print(self.kernels_var)
        print("calculate merged location")
        s_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * self.kernels_s[kernel_min_ind] + h_new * s_new)
        print("calculate merged std")
        std_merge = (1.0/h_merge)*(self.kernels_h[kernel_min_ind] * (np.square(self.kernels_var[kernel_min_ind]) + np.square(self.kernels_s[kernel_min_ind])) + h_new * (np.square(var_new) + np.square(s_new))) - np.square(s_merge)
        self.kernels_h[kernel_min_ind] = h_merge
        self.kernels_var[kernel_min_ind] = np.sqrt(std_merge)
        self.kernels_s[kernel_min_ind] = s_merge

    #add new kernel to lists
    def add_kernel_to_compressed(self,h_new, s_new, var_new):
        self.kernels_h = self.kernels_h + [h_new]
        self.kernels_s = self.kernels_s + [s_new]
        self.kernels_var = self.kernels_var + [var_new]

    #check whether merging or adding should be fulfilled
    def compression_check(self,h_new,s_new,var_new):
        dist_values = self.calc_min_dist(s_new)
        print("minimal distance is: ", dist_values[1])
        if dist_values[1] < self.threshold_kde:
            print("merge with: ", dist_values[0])
            self.merge_kernels(dist_values[0], h_new, s_new, var_new)
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