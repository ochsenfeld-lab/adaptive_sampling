import sys
import pytest
import torch

from adaptive_sampling.colvars.path_cv import PathCV
from adaptive_sampling.units import BOHR_to_ANGSTROM

sys.path.append("resources/")

@pytest.mark.parametrize(
    "input, path, bounds", [
        (
            "resources/path.xyz", 
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ], 
            [
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ]
        )
    ]
)
def test_init(input, path, bounds):
    
    path = [path[i] / BOHR_to_ANGSTROM for i in range(len(path))]
    bounds = [bounds[i] / BOHR_to_ANGSTROM for i in range(len(bounds))]
    
    cv = PathCV(guess_path=input)

    assert cv.nnodes == len(path)
    assert cv.natoms == int(len(path[0])/3)
    assert torch.equal(cv.path[0], path[0])
    assert torch.equal(cv.path[1], path[1])
    assert torch.equal(cv.boundary_nodes[0], bounds[0])
    assert torch.equal(cv.boundary_nodes[1], bounds[1])
    
@pytest.mark.parametrize(
    "input, coords1, coords2", [
        (
            "resources/path.xyz", 
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    ]
)
def test_calculate(input, coords1, coords2):
    
    coords1 = coords1 / BOHR_to_ANGSTROM
    coords2 = coords2 / BOHR_to_ANGSTROM 
    
    cv = PathCV(guess_path=input)

    cv1 = cv.calculate(coords1)
    cv2 = cv.calculate(coords2) 
    assert cv1 == 0
    assert cv2 == 1.5
