import numpy
import torch
from math import isclose

from adaptive_sampling.colvars.graph_cv import GRAPH_CV
from adaptive_sampling import units

def test_sprint_A_matrix():

    the_sprint = GRAPH_CV(
        atom_indices=[0, 1, 2, 3],
        atom_types=["H", "H", "H", "H"],
        N=30,
        requires_grad=False,
        parallel=False,
    )

    z = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ]
    ) / units.BOHR_to_ANGSTROM

    _ = the_sprint.calc(z)
    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    print(A, the_sprint.A)
    assert torch.allclose(the_sprint.A, A, atol=0.1)


def test_sprint_cv():

    the_sprint = GRAPH_CV(
        atom_indices=[0, 1, 2, 3],
        atom_types=["C", "C", "C", "C"],
        N=6,
        requires_grad=False,
        parallel=False,
    )

    z = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ]
    ) / units.BOHR_to_ANGSTROM

    cv = the_sprint.calc(z)
    assert isclose(float(cv), 1.875, abs_tol=1e-2)


def test_sprint_grad():

    the_sprint = GRAPH_CV(
        atom_indices=[0, 1, 2, 3],
        atom_types=["C", "O", "N", "C"],
        N=30,
        requires_grad=True,
        parallel=False,
    )

    z = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    ) / units.BOHR_to_ANGSTROM

    _ = the_sprint.calc(z)

    reference_gradient = numpy.array(
        [
            [0.00145974, 0.0, 0.0],
            [0.0009077, 0.0, 0.0],
            [-0.0009077, 0.0, 0.0],
            [-0.00145974, 0.0, 0.0],
        ]
    )

    assert numpy.allclose(the_sprint.gradient, reference_gradient, atol=0.01)


def test_sprint_A_matrix_parallel():

    the_sprint = GRAPH_CV(
        atom_indices=[0, 1, 2, 3],
        atom_types=["C", "C", "C", "C"],
        N=30,
        requires_grad=False,
        parallel=False,
    )

    z = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ]
    ) / units.BOHR_to_ANGSTROM

    _ = the_sprint.calc(z)
    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert torch.allclose(the_sprint.A, A, atol=0.1)
