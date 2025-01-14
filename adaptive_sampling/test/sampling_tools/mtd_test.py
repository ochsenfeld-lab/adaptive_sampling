import numpy as np
import pytest as pytest
from adaptive_sampling.sampling_tools.metadynamics import *
from adaptive_sampling.interface.sampling_data import SamplingData
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
    masses = [2., 1., 1., 10.]
    coords = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.]
    return MD(masses, coords)


def test_one_hill_1D():

    the_bias = WTM(
        four_particles(),
        [["distance", [0, 1], 0, 3, 0.001]],
        hill_height=1.0,
        hill_std=np.ones(1),
        hill_drop_freq=1,
        periodicity=None,
        bias_factor=np.inf,
        kinetics=True,
    )
    the_bias.step_bias(traj_file=None, out_file=None, restart_file=None)

    # first kernel droped
    assert the_bias.hills_center == np.array([1.0])
    assert the_bias.hills_height == np.array([1.0])/atomic_to_kJmol
    assert the_bias.hills_std == np.array([1.0])/BOHR_to_ANGSTROM
    assert the_bias.well_tempered == False

    # value of Gaussian at center 
    assert np.sum(the_bias.calc_hills([1.0]))*atomic_to_kJmol == pytest.approx(1.0, abs=1e-4)
    
    # value of Gaussian at distance 1 std    
    assert np.sum(the_bias.calc_hills([the_bias.hills_center[0]+the_bias.hills_std[0]]))*atomic_to_kJmol == pytest.approx(0.6065, abs=1e-4)

    # value of Gaussian at center from grid
    bin = the_bias.get_index([the_bias.hills_center[0]])
    assert the_bias.metapot[bin[1], bin[0]]*atomic_to_kJmol == pytest.approx(1.0, abs=1e-3)

    # value of Gaussian at distance 1 std from grid
    bin = the_bias.get_index([the_bias.hills_center[0]+the_bias.hills_std[0]])
    assert the_bias.metapot[bin[1], bin[0]]*atomic_to_kJmol == pytest.approx(0.6065, abs=1e-3)


def test_one_hill_2D():

    the_bias = WTM(
        four_particles(),
        [["distance", [0, 1], 0, 2., 0.01],
         ["distance", [2, 3], 0, 2., 0.01]],
        hill_height=1.0,
        hill_std=np.ones(2),
        hill_drop_freq=1,
        periodicity=None,
        bias_factor=np.inf,
        kinetics=True,
    )
    the_bias.step_bias(traj_file=None, out_file=None, restart_file=None)

    # first kernel droped
    assert np.all(the_bias.hills_center == np.array([1.0, 1.0]))
    assert the_bias.hills_height == np.array([1.0])/atomic_to_kJmol
    assert np.all(the_bias.hills_std == np.array([1.0,1.0])/BOHR_to_ANGSTROM)
    assert the_bias.well_tempered == False

    # value of Gaussian at center 
    assert np.sum(the_bias.calc_hills(the_bias.hills_center[0]))*atomic_to_kJmol == pytest.approx(1.0, abs=1e-4)
    
    # value of Gaussian at distance 1 std    
    std_1d  = np.array([the_bias.hills_std[0][0], 0.0])
    assert np.sum(the_bias.calc_hills([the_bias.hills_center[0]+std_1d]))*atomic_to_kJmol == pytest.approx(0.6065, abs=1e-4)

    # value of Gaussian at center from grid
    bin = the_bias.get_index(the_bias.hills_center[0])
    assert the_bias.metapot[bin[1], bin[0]]*atomic_to_kJmol == pytest.approx(1.0, abs=1e-2)

    # value of Gaussian at distance 1 std from grid
    bin = the_bias.get_index(the_bias.hills_center[0]+std_1d)
    assert the_bias.metapot[bin[1], bin[0]]*atomic_to_kJmol == pytest.approx(0.6065, abs=1e-2)