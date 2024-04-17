from adaptive_sampling.sampling_tools.utils import gaussian_calc
from adaptive_sampling.sampling_tools.utils import distance_calc
from adaptive_sampling.sampling_tools.utils import correct_periodicity
from adaptive_sampling.sampling_tools.opes import OPES
from adaptive_sampling.interface.sampling_data import SamplingData
import numpy as np

def test_gaussian_calc():
    s = np.array([3,3])
    kernel_var = np.array(1)
    s_new = np.array([3,3])
    periodicity: list = [None]
    gauss = np.array(gaussian_calc(s,kernel_var,s_new,periodicity))
    assert gauss == 1.0

def test_distance_calc():
    periodicity: list = [None]
    distance = distance_calc(np.array([2.0]),np.array([1.0]),np.array([1.0]),periodicity)
    assert distance == 1.0

class MD:
    def __init__(self, mass, coords):
        self.masses = np.array(mass)
        self.coords = np.array(coords)
        self.natoms = len(mass)
        self.forces = np.zeros(3 * self.natoms)

    def get_sampling_data(self) -> SamplingData:
        return SamplingData(
            self.masses,
            self.coords,
            np.zeros_like(self.coords),
            0.0,
            0.0,
            self.natoms,
            0,
            0.0,
        )
    
def four_particles():
    masses = [2, 1, 1, 10]
    coords = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    return MD(masses, coords)


def four_particles2():
    masses = [1, 3, 1, 10]
    coords = [0, 0, 0, 3, 0, 0, 1, 0, 0, 1, 1, 1]
    return MD(masses, coords)

#print("1D Test")
#OPES1 = OPES(four_particles(), [["distance", [0,1], 0,0,0]])
#OPES1.show_kernel_lists()
#OPES1.compression_check(1.,np.array([1.0]),np.array([1.0]))
#OPES1.show_kernel_lists()
#print("done cycle 1")
#OPES1.compression_check(1.,np.array([2.0]),np.array([1.0]))
#OPES1.show_kernel_lists()
#print("done cycle 2")
#OPES1.compression_check(1.0,np.array([2.5]),np.array([1.0]))
#OPES1.show_kernel_lists()
#print("done cycle 3")
#print("1D Test successful")

def test_kde_1D_2():
    OPES_test1_2 = OPES(four_particles(), [["distance", [0,1], 0,0,0]], kernel_location=np.ones(1),kernel_var=np.ones(1))
    OPES_test1_2.compression_check(1.,np.array([1.0]),np.array([1.0]))
    OPES_test1_2.compression_check(1.,np.array([2.0]),np.array([1.0]))
    assert OPES_test1_2.kernels_h == [2.0,1.]
    assert OPES_test1_2.kernels_s == [1.,2.]
    assert OPES_test1_2.kernels_var == [1.,1.]

def test_kde_1D():
    OPES_test1 = OPES(four_particles(), [["distance", [0,1], 0,0,0]], kernel_location=np.ones(1),kernel_var=np.ones(1))
    OPES_test1.compression_check(1.,np.array([1.0]),np.array([1.0]))
    OPES_test1.compression_check(1.,np.array([2.0]),np.array([1.0]))
    OPES_test1.compression_check(1.0,np.array([2.5]),np.array([1.0]))
    assert OPES_test1.kernels_h == [2.0,2.]
    assert OPES_test1.kernels_s == [1.,2.25]
    assert OPES_test1.kernels_var == [1.,np.sqrt(1.0625)]

def test_kde_2D():
    OPES_test2 = OPES(four_particles(), [["distance", [0,1], 0,0,0]], kernel_location=np.ones(2),kernel_var=np.ones(2))
    OPES_test2.compression_check(1.,np.array([1.0,1.0]),np.array([1.0,1.0]))
    OPES_test2.compression_check(1.,np.array([2.0,1.0]),np.array([1.0,1.0]))
    OPES_test2.compression_check(1.0,np.array([2.5,1.0]),np.array([1.0,1.0]))
    assert OPES_test2.kernels_h == [2.0,2.0]
    assert np.allclose(np.array(OPES_test2.kernels_s), np.array([np.array([1.0,1.0]),np.array([2.25,1.0])]))
    assert np.allclose(np.array(OPES_test2.kernels_var), np.array([np.array([1.0,1.0]),np.array([np.sqrt(1.0625),1.0])]))

print("2D Test")
OPES2 = OPES(four_particles(), [["distance", [0,1], 0,0,0]])
OPES2.show_kernel_lists()
OPES2.compression_check(np.array([1.0,1.0]),np.array([0.1,0.1]))
OPES2.show_kernel_lists()
print("done cycle 1")
OPES2.compression_check(np.array([2.0,1.0]),np.array([0.1,0.5]))
OPES2.show_kernel_lists()
print("done cycle 2")
#OPES2.compression_check(np.array([1.51,1.0]),np.array([0.1,0.1]))
#OPES2.show_kernel_lists()
print("done cycle 3")
OPES2.compression_check(np.array([3.25,1.4]),np.array([0.1,0.1]))
OPES2.show_kernel_lists()
print("done cycle 4")
#OPES2.compression_check(np.array([2.75,1.0]),np.array([0.1,0.1]))
#OPES2.show_kernel_lists()
print("done cycle 5")
OPES2.calc_prob_dist(np.array([2.5,1.0]))
OPES2.calculate_potential(np.array([2.5,1.0]))
print("2D Test successful")

