import torch
import pytest
import numpy as np
from adaptive_sampling.colvars.colvars import CV
from adaptive_sampling.colvars.utils import *
from adaptive_sampling.interface.sampling_data import SamplingData
from adaptive_sampling.units import *


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


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [9, 9, 7],
                    [9, 9, 8],
                    [9, 9, 9],
                ]
            ),
            torch.tensor(
                [
                    [30.50347534, -20.16089091, -7.42752623],
                    [30.77704903, -21.02339348, -7.27823201],
                    [31.3215374, -21.99452332, -7.15703548],
                    [42.05988643, -23.50924264, -15.59516355],
                    [42.27217891, -24.36478643, -15.59064995],
                    [42.66080502, -25.27318759, -15.386241],
                ]
            ),
            26.6020,
            0.0565,
        )
    ],
)
def test_kabsch(a, b, expected_nofit, expected_fit):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = get_rmsd(a, b)
    rmsd_kabsch = kabsch_rmsd(a, b)
    assert float(rmsd_nofit) == pytest.approx(expected_nofit, rel=1.0e-3)
    assert float(rmsd_kabsch) == pytest.approx(expected_fit, rel=1.0e-3)


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ),
            0.707,
            0.000,
        )
    ],
)
def test_kabsch_1(a, b, expected_nofit, expected_fit):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = get_rmsd(a, b)
    rmsd_kabsch = kabsch_rmsd(a, b)

    assert float(rmsd_nofit) == pytest.approx(expected_nofit, abs=1.0e-3)
    assert float(rmsd_kabsch) == pytest.approx(expected_fit, abs=1.0e-3)


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [9, 9, 7],
                    [9, 9, 8],
                    [9, 9, 9],
                ]
            ),
            torch.tensor(
                [
                    [30.50347534, -20.16089091, -7.42752623],
                    [30.77704903, -21.02339348, -7.27823201],
                    [31.3215374, -21.99452332, -7.15703548],
                    [42.05988643, -23.50924264, -15.59516355],
                    [42.27217891, -24.36478643, -15.59064995],
                    [42.66080502, -25.27318759, -15.386241],
                ]
            ),
            26.6020,
            0.0565,
        )
    ],
)
def test_quaternion(a, b, expected_nofit, expected_fit):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = get_rmsd(a, b)
    rmsd_quaternion = quaternion_rmsd(a, b)

    assert float(rmsd_nofit) == pytest.approx(expected_nofit, rel=1.0e-3)
    assert float(rmsd_quaternion) == pytest.approx(expected_fit, rel=1.0e-3)


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ),
            0.707,
            0.000,
        )
    ],
)
def test_quaternion_1(a, b, expected_nofit, expected_fit):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = get_rmsd(a, b)
    rmsd_quaternion = quaternion_rmsd(a, b)

    assert float(rmsd_nofit) == pytest.approx(expected_nofit, abs=1.0e-3)
    assert float(rmsd_quaternion) == pytest.approx(expected_fit, abs=1.0e-3)


@pytest.mark.parametrize(
    "file, coords",
    [
        (
            "resources/m1.xyz",
            torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]) / BOHR_to_ANGSTROM,
        )
    ],
)
def test_read_xyz(file, coords):
    xyz = read_xyz(file)
    assert torch.allclose(xyz, coords.float())


@pytest.mark.parametrize(
    "coords1, coords2, rmsd_fit", [("resources/m1.xyz", "resources/m2.xyz", 0.000)]
)
def test_rmsd(coords1, coords2, rmsd_fit):
    xyz1 = read_xyz(coords1)
    the_md = MD([1, 2, 3, 4], xyz1.numpy())
    the_cv = CV(the_md, requires_grad=False)
    f1 = the_cv.get_cv("rmsd", coords2, method="kabsch")
    f2 = the_cv.get_cv("rmsd", coords2, method="quaternion")
    assert f1 == pytest.approx(rmsd_fit, abs=1e-3)
    assert f2 == pytest.approx(rmsd_fit, abs=1e-3)
    #assert np.allclose(grad1, grad2)
