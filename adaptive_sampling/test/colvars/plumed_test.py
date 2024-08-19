import numpy as np
from adaptive_sampling.colvars.plumed_cv import PLUMED_CV
from adaptive_sampling.units import *


def test_init_plumed():
    p = PLUMED_CV("DISTANCE ATOMS=1,2", 3)
    assert np.allclose(p.masses, np.ones(3))
    assert p.natoms == 3


def test_grad_plumed():
    p = PLUMED_CV("DISTANCE ATOMS=1,2", 3)
    coords = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]])
    cv = p.calc(coords)

    d = coords[1] - coords[0]
    grad = np.zeros_like(coords)
    grad[0] = -d / np.linalg.norm(d)
    grad[1] = d / np.linalg.norm(d)
    assert np.allclose(2.0, cv)
    assert np.allclose(grad[:2], p.gradient[:2])
