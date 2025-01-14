import numpy as np
import pytest as pytest
from adaptive_sampling.sampling_tools.opes import *
from adaptive_sampling.interface.sampling_data import SamplingData
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *


class MD:
    def __init__(self, mass, coords):
        self.mass = np.array(mass)
        self.coords = np.array(coords)
        self.natoms = len(mass)
        self.forces = np.zeros(3 * self.natoms)

    def get_sampling_data(self) -> SamplingData:
        return SamplingData(
            self.mass,
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


the_bias = OPES(
    four_particles(),
    [["distance", [0, 1], 0, 10, 1]],
    # kernel_location=np.ones(1),
    kernel_std=np.ones(1),
    adaptive_sigma=False,
    unbiased_time=10,
    fixed_sigma=False,
    explore=False,
    periodicity=None,
    output_freq=1000,
    equil_temp=300.0,
    energy_barr=20,
    merge_threshold=1.0,
    approximate_norm=True,
    exact_norm=True,
    verbose=True,
    recursion_merge=True,
    update_freq=100,
    f_conf=1000.0,
    bias_factor=None,
    kinetics=True,
)

the_bias.delta_kernel_height = []
the_bias.delta_kernel_center = []
the_bias.delta_kernel_sigma = []

the_bias.kernel_center.append(np.array([1.5]))
the_bias.kernel_height.append(1.0)
the_bias.kernel_std.append(np.array([0.5]))

the_bias.kernel_center.append(np.array([3.0]))
the_bias.kernel_height.append(1.0)
the_bias.kernel_std.append(np.array([0.5]))


# Unit Tests
def test_get_val_gaussian_on_first_kernel_center():
    assert the_bias.calc_gaussians(1.5)[0] == pytest.approx(1.0)


def test_get_val_gaussian_on_two_kernel():
    assert np.sum(the_bias.calc_gaussians(2.5)) == pytest.approx(0.74186594)


def test_calc_min_dist():
    assert the_bias.calc_min_dist(1.75)[0] == 0
    assert the_bias.calc_min_dist(1.75)[1] == pytest.approx(0.5)


def test_calc_pot():
    assert the_bias.calc_potential(1.0) == pytest.approx(0.00666, abs=1e-4)


def test_calculate_forces():
    assert the_bias.calc_forces(
        1.0, 0.2
    ) == pytest.approx(0.00016, abs=1e-5)


def test_calc_norm_factor_exact():
    assert the_bias.calc_norm_factor(approximate=False) == pytest.approx(
        3069.313587526749
    )


def test_add_kernel_for_new_kernel_out_of_merging_threshold():
    the_bias.add_kernel(1.0, np.array([4.5]), np.array([1.0]))
    assert the_bias.kernel_center == [np.array([1.5]), np.array([3.0]), np.array([4.5])]
    assert the_bias.kernel_height == [1.0, 1.0, 1.0]
    assert the_bias.kernel_std == [np.array([0.5]), np.array([0.5]), np.array([1.0])]


def test_add_kernel_for_new_kernel_in_merging_threshold_of_second():
    the_bias.add_kernel(1.0, np.array([2.75]), np.array([1.0]))
    assert the_bias.kernel_center == [
        np.array([1.5]),
        np.array([4.5]),
        np.array([2.875]),
    ]
    assert the_bias.kernel_height == [1.0, 1.0, 2.0]
    assert the_bias.kernel_std == [
        np.array([0.5]),
        np.array([1.0]),
        np.array([pytest.approx(0.80039053)]),
    ]
