import sys
import pytest
import torch
from math import isclose

from adaptive_sampling.colvars.path_cv import PathCV
from adaptive_sampling.units import BOHR_to_ANGSTROM

sys.path.append("resources/")

@pytest.mark.parametrize(
    "input, path, bounds", [
        (
            "resources/path.xyz", 
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ], 
            [
                torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                torch.tensor([7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        )
    ]
)
def test_init(input, path, bounds):
    path = [path[i] / BOHR_to_ANGSTROM for i in range(len(path))]
    bounds = [bounds[i] / BOHR_to_ANGSTROM for i in range(len(bounds))]
    
    cv = PathCV(guess_path=input, metric="abs_distance")
    
    assert cv.nnodes == len(path)
    assert cv.natoms == int(len(path[0])/3)
    assert torch.allclose(cv.path[0], path[0])
    assert torch.allclose(cv.path[1], path[1])
    assert torch.allclose(cv.path[2], path[2])
    assert torch.allclose(cv.boundary_nodes[0], bounds[0])
    assert torch.allclose(cv.boundary_nodes[1], bounds[1])
    
@pytest.mark.parametrize(
    "input, coords1, coords2, coords3", [
        (
            "resources/path.xyz", 
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ]
)
def test_calculate_gpath(input, coords1, coords2, coords3):
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM 
    coords3 = coords3 / BOHR_to_ANGSTROM

    cv = PathCV(guess_path=input, metric="RMSD")

    cv1 = cv.calculate_gpath(coords1)
    cv2 = cv.calculate_gpath(coords2) 
    cv3 = cv.calculate_gpath(coords3)
    assert isclose(float(cv1), float(0.0), abs_tol=1e-1)
    assert isclose(float(cv2), float(0.7), abs_tol=1e-1)
    assert isclose(float(cv3), float(1.0), abs_tol=1e-1)

@pytest.mark.parametrize(
    "input, coords1, coords2, coords3", [
        (
            "resources/path.xyz", 
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([6.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ]
)
def test_calculate_path(input, coords1, coords2, coords3):
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM 
    coords3 = coords3 / BOHR_to_ANGSTROM

    cv = PathCV(guess_path=input, metric="RMSD", adaptive=True)

    cv1 = cv.calculate_path(coords1)
    cv2 = cv.calculate_path(coords2) 
    cv3 = cv.calculate_path(coords3)
    assert isclose(float(cv1), float(0.1), abs_tol=1e-1)
    assert isclose(float(cv2), float(0.5), abs_tol=1e-1)
    assert isclose(float(cv3), float(0.9), abs_tol=1e-1)

@pytest.mark.parametrize(
    "input, coords1, coords2", [
        (
            "resources/path.xyz", 
            torch.tensor([3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ]
)
def test_projection_point_on_path(input, coords1, coords2):
    coords1 /= BOHR_to_ANGSTROM
    coords2 /= BOHR_to_ANGSTROM
    cv = PathCV(guess_path=input)
    rmsds = cv._get_rmsds_to_path(coords1)
    _, q = cv._get_closest_nodes(coords1, rmsds)
    cv1 = cv._project_coords_on_path(coords1, q)
    assert torch.allclose(cv1, coords2)

@pytest.mark.parametrize(
    "input, coords, zmatrix", [
        (
            "resources/path.xyz", 
            torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0]),
            torch.tensor([
                [0.0, 0.0, 0.0], 
                [1.0, 0.0, 0.0], 
                [1.0, 1.5708, 0.0]
            ])
        )
    ]
)
def test_internals(input, coords, zmatrix):
    cv = PathCV(guess_path=input, metric="internal")
    cv.active = [0,1,2]
    zm = cv._cartesians_to_internals(coords)
    assert torch.allclose(zm, zmatrix)

@pytest.mark.parametrize(
    "input, coords, reference", [
        (
            "resources/path.xyz", 
            torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0]),
        )
    ]
)
def test_internal_rmsd(input, coords, reference):
    cv = PathCV(guess_path=input, metric="internal")
    cv.active = [0,1,2]
    zm = cv._cartesians_to_internals(reference)
    print(zm)

    rmsd = cv._rmsd(coords, reference)
    assert isclose(0.5, rmsd, abs_tol=1.e-1)
