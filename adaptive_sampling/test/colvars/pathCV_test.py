import sys
import pytest
import torch
from math import isclose

from adaptive_sampling.colvars.path_cv import PathCV
from adaptive_sampling.colvars.utils import *
from adaptive_sampling.units import BOHR_to_ANGSTROM

sys.path.append("resources/")


@pytest.mark.parametrize(
    "input, path, bounds",
    [
        (
            "resources/path.xyz",
            [
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[7.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ],
        )
    ],
)
def test_init_cartesian(input, path, bounds):
    path = [path[i] / BOHR_to_ANGSTROM for i in range(len(path))]
    bounds = [bounds[i] / BOHR_to_ANGSTROM for i in range(len(bounds))]

    cv = PathCV(
        guess_path=input,
        metric="RMSD",
        smooth_damping=0.0,
        coordinate_system="Cartesian",
    )
    assert cv.nnodes == len(path)
    assert torch.allclose(cv.path[0], path[0], atol=1.0e-1)
    assert torch.allclose(cv.path[1], path[1], atol=1.0e-1)
    assert torch.allclose(cv.path[2], path[2], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[0], bounds[0], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[1], bounds[1], atol=1.0e-1)


@pytest.mark.parametrize(
    "input, path, bounds",
    [
        (
            "resources/path.xyz",
            [
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
            ],
        )
    ],
)
def test_init_zmatrix(input, path, bounds):
    path = [path[i] / BOHR_to_ANGSTROM for i in range(len(path))]
    bounds = [bounds[i] / BOHR_to_ANGSTROM for i in range(len(bounds))]

    cv = PathCV(
        guess_path=input, metric="RMSD", smooth_damping=0.0, coordinate_system="ZMatrix"
    )

    assert cv.nnodes == len(path)
    assert torch.allclose(cv.path[0], path[0], atol=1.0e-1)
    assert torch.allclose(cv.path[1], path[1], atol=1.0e-1)
    assert torch.allclose(cv.path[2], path[2], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[0], bounds[0], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[1], bounds[1], atol=1.0e-1)


@pytest.mark.parametrize(
    "input, path, bounds",
    [
        (
            "resources/path.xyz",
            [
                torch.tensor([1.0]),
                torch.tensor([3.0]),
                torch.tensor([5.0]),
            ],
            [
                torch.tensor([-1.0]),
                torch.tensor([7.0]),
            ],
        )
    ],
)
def test_init_cv_space(input, path, bounds):
    path = [path[i] / BOHR_to_ANGSTROM for i in range(len(path))]
    bounds = [bounds[i] / BOHR_to_ANGSTROM for i in range(len(bounds))]

    cv = PathCV(
        guess_path=input,
        metric="RMSD",
        smooth_damping=0.0,
        coordinate_system="cv_space",
        active=[["distance", [0, 1]]],
    )

    assert cv.nnodes == len(path)
    assert torch.allclose(cv.path[0], path[0], atol=1.0e-1)
    assert torch.allclose(cv.path[1], path[1], atol=1.0e-1)
    assert torch.allclose(cv.path[2], path[2], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[0], bounds[0], atol=1.0e-1)
    assert torch.allclose(cv.boundary_nodes[1], bounds[1], atol=1.0e-1)


@pytest.mark.parametrize(
    "input, coords1, coords2, coords3, coords4, coords5",
    [
        (
            "resources/path.xyz",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ],
)
def test_calculate_gpath(input, coords1, coords2, coords3, coords4, coords5):
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM
    coords3 = coords3 / BOHR_to_ANGSTROM
    coords4 = coords4 / BOHR_to_ANGSTROM
    coords5 = coords5 / BOHR_to_ANGSTROM

    cv = PathCV(
        guess_path=input,
        metric="RMSD",
        smooth_damping=0.0,
        coordinate_system="cv_space",
        active=[["distance", [0, 1]]],
    )
    cv1 = cv.calculate_gpath(coords1)
    cv2 = cv.calculate_gpath(coords2)
    cv3 = cv.calculate_gpath(coords3)
    cv4 = cv.calculate_gpath(coords4)
    cv5 = cv.calculate_gpath(coords5)
    assert isclose(float(cv1), float(0.0), abs_tol=1e-2)
    assert isclose(float(cv2), float(0.25), abs_tol=1e-2)
    assert isclose(float(cv3), float(0.5), abs_tol=1e-2)
    assert isclose(float(cv4), float(0.75), abs_tol=1e-2)
    assert isclose(float(cv5), float(1.0), abs_tol=1e-2)


@pytest.mark.parametrize(
    "input, coords1, coords2, coords3, coords4, coords5",
    [
        (
            "resources/path.xyz",
            torch.tensor([1.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ],
)
def test_calculate_path(input, coords1, coords2, coords3, coords4, coords5):
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM
    coords3 = coords3 / BOHR_to_ANGSTROM
    coords4 = coords4 / BOHR_to_ANGSTROM
    coords5 = coords5 / BOHR_to_ANGSTROM

    cv = PathCV(
        guess_path=input,
        metric="rmsd",
        smooth_damping=0.0,
        coordinate_system="Cartesian",
        active=[0, 1],
    )

    cv1 = cv.calculate_path(coords1)
    cv2 = cv.calculate_path(coords2)
    cv3 = cv.calculate_path(coords3)
    cv4 = cv.calculate_path(coords4)
    cv5 = cv.calculate_path(coords5)
    assert isclose(float(cv1), float(0.0), abs_tol=1e-2)
    assert isclose(float(cv2), float(0.25), abs_tol=1e-1)
    assert isclose(float(cv3), float(0.5), abs_tol=1e-2)
    assert isclose(float(cv4), float(0.75), abs_tol=1e-1)
    assert isclose(float(cv5), float(1.0), abs_tol=1e-2)


@pytest.mark.parametrize(
    "input, coords1, coords2, coords3, coords4, coords5",
    [
        (
            "resources/path.xyz",
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([4.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([5.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ],
)
def test_calculate_path_z(input, coords1, coords2, coords3, coords4, coords5):
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM
    coords3 = coords3 / BOHR_to_ANGSTROM
    coords4 = coords4 / BOHR_to_ANGSTROM
    coords5 = coords5 / BOHR_to_ANGSTROM

    coords1.requires_grad = True
    coords2.requires_grad = True
    coords3.requires_grad = True
    coords4.requires_grad = True
    coords5.requires_grad = True

    cv = PathCV(
        guess_path=input,
        metric="msd",
        smooth_damping=0.0,
        coordinate_system="Cartesian",
        active=[0, 1],
        requires_z=True,
    )

    cv.calculate_path(coords1)
    cv1 = cv.path_z
    cv.calculate_path(coords2)
    cv2 = cv.path_z
    cv.calculate_path(coords3)
    cv3 = cv.path_z
    cv.calculate_path(coords4)
    cv4 = cv.path_z
    cv.calculate_path(coords5)
    cv5 = cv.path_z
    assert isclose(float(cv1), float(1.75), abs_tol=1e-2)
    assert isclose(float(cv2), float(1.06), abs_tol=1e-2)
    assert isclose(float(cv3), float(1.72), abs_tol=1e-2)
    assert isclose(float(cv4), float(1.06), abs_tol=1e-2)
    assert isclose(float(cv5), float(1.75), abs_tol=1e-2)


@pytest.mark.parametrize(
    "input, coords1, coords2",
    [
        (
            "resources/path.xyz",
            torch.tensor([3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ],
)
def test_projection_point_on_path(input, coords1, coords2):
    coords1 = coords1.view(2, 3) / BOHR_to_ANGSTROM
    coords2 = coords2.view(2, 3) / BOHR_to_ANGSTROM
    cv = PathCV(guess_path=input, metric="RMSD")
    rmsds = cv._get_distance_to_path(coords1)
    idx_min = cv._get_closest_nodes(coords1, rmsds)
    nodes = [cv.path[idx_min[0]], cv.path[idx_min[1]]]
    cv1 = cv._project_coords_on_line(coords1, nodes)
    assert torch.allclose(cv1, coords2)


@pytest.mark.parametrize(
    "coords, zmatrix",
    [
        (
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            torch.tensor(
                [
                    [0.0, 0.0000, 0.0000],
                    [1.0, 0.0000, 0.0000],
                    [1.0, 1.5708, 0.0000],
                    [1.0, 1.5708, 1.5708],
                ]
            ),
        )
    ],
)
def test_internals(coords, zmatrix):
    zm = cartesians_to_internals(coords)
    assert torch.allclose(zm, zmatrix)


@pytest.mark.parametrize(
    "input, coords",
    [
        (
            "resources/path.xyz",
            torch.tensor([2.0]),
        )
    ],
)
def test_selected_rmsd(input, coords):
    coords /= BOHR_to_ANGSTROM

    cv = PathCV(
        guess_path=input,
        metric="RMSD",
        coordinate_system="cv_space",
        active=[["distance", [0, 1]]],
    )

    rmsd0 = get_rmsd(coords, cv.path[0]) * BOHR_to_ANGSTROM
    rmsd1 = get_rmsd(coords, cv.path[1]) * BOHR_to_ANGSTROM
    rmsd2 = get_rmsd(coords, cv.path[2]) * BOHR_to_ANGSTROM

    assert isclose(1.0, rmsd0, abs_tol=1.0e-1)
    assert isclose(1.0, rmsd1, abs_tol=1.0e-1)
    assert isclose(3.0, rmsd2, abs_tol=1.0e-1)
