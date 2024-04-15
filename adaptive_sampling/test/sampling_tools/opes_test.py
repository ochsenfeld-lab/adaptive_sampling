
from adaptive_sampling.sampling_tools.utils import gaussian_calc
from adaptive_sampling.sampling_tools.utils import distance_calc
from adaptive_sampling.sampling_tools.opes import OPES
import numpy as np

def test_gaussian_calc():
    h = 1
    s = np.array([3,3])
    kernel_var = np.array(1)
    s_new = np.array([3,3])
    gauss = np.array(gaussian_calc(h,s,kernel_var,s_new))
    assert gauss == 1.0

def test_distance_calc():
    distance = distance_calc(2,1,1)
    assert distance == 1.0


OPES1 = OPES()

OPES1.compression_check(1,1,1)
