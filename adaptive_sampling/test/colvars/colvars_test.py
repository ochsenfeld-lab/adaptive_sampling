import numpy as np
from adaptive_sampling.colvars import CV
from adaptive_sampling.interface.sampling_data import SamplingData


class MD:
    def __init__(self, mass, coords):
        self.masses = np.array(mass)
        self.coords = np.array(coords)
        self.natoms = len(mass)
        self.forces = np.zeros(3 * self.natoms)

    def get_sampling_data(self) -> SamplingData:
        return SamplingData(self.masses, self.coords, np.zeros_like(self.coords), 0.0, 0.0, self.natoms, 0, 0.0)


def four_particles():
    masses = [2, 1, 1, 10]
    coords = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    return MD(masses, coords)

def four_particles2():
    masses = [1, 3, 1, 10]
    coords = [0, 0, 0, 3, 0, 0, 1, 0, 0, 1, 1, 1]
    return MD(masses, coords)


def test_distance():
    cv = CV(four_particles(), requires_grad=True)
    f, grad = cv.get_cv("distance", [0, 1])
    assert f == 1
    assert (grad == np.asarray([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all()


def test_distance_groups():
    cv = CV(four_particles2(), requires_grad=True)
    f = cv.distance([[0, 1], 2])
    assert f == 1.25
    assert (cv.gradient == np.asarray([0.25, 0, 0, 0.75, 0, 0, -1, 0, 0, 0, 0, 0])).all()


def test_angle():
    cv = CV(four_particles(), requires_grad=True)
    f, grad = cv.get_cv("angle", [0, 1, 3])
    f /= np.pi / 180.0
    assert int(f) == 90
    assert (grad.sum() < 1e-7) 


def test_torsion():
    cv = CV(four_particles(), requires_grad=False)
    f = cv.get_cv("torsion", [0, 1, 2, 3])
    f /= np.pi / 180.0
    assert int(f) == 90


def test_linear_combination():
    cv = CV(four_particles(), requires_grad=True)
    f = cv.linear_combination([[1.0, [2, 3]], [2.0, [0, 1]]])
    print(cv.gradient)
    assert f == 3.0
    assert (cv.gradient == np.asarray([-2, 0, 0, 2, 0, 0, 0, 0, -1, 0, 0, 1])).all()
