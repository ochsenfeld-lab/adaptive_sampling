from adaptive_sampling.sampling_tools.utils import gaussian_calc
from adaptive_sampling.sampling_tools.utils import distance_calc
from adaptive_sampling.sampling_tools.utils import correct_periodicity
from adaptive_sampling.sampling_tools.opes import OPES
from adaptive_sampling.interface.sampling_data import SamplingData
import numpy as np

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

#print("2D Test")
#OPES2 = OPES(four_particles(), [["distance", [0,1], 0,0,0]])
#OPES2.update_kde(np.array([2.0,1.5]))
#print("2D Test successful")

print("initiate 1D Test")
OPES1 = OPES(0.1, four_particles(), [["distance", [0,1], 0,0,0]], merge_kernels=True, approximate_norm=False,verbose=True)
OPES1.update_kde(np.array([1.1]))
OPES1.md_step = 1
print(OPES1.md_step)
#OPES1.update_kde(np.array([1.25]))
#OPES1.update_kde(np.array([1.11]))
#OPES1.update_kde(np.array([1.124]))
#OPES1.update_kde(np.array([1.214]))
#OPES1.update_kde(np.array([1.123]))
#OPES1.update_kde(np.array([1.342]))
OPES1.step_bias()
print("1D Test successful")
