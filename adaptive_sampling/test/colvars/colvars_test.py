import numpy as np
from adaptive_sampling.colvars import CV


class MD:
    def __init__(self, mass, coords):
        self.masses = np.array(mass)
        self.coords = np.array(coords)
        self.natoms = len(mass)
        self.forces = np.zeros(3 * self.natoms)


def four_particles():
    masses = [2, 1, 1, 10]
    coords = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    return MD(masses, coords)


the_mol = four_particles()
cv = CV(the_mol, requires_grad=True)
f, grad = cv.get_cv("distance", [0, 2])
print("\nDistance: ", f)
print("Gradient:\n", grad)

the_mol.coords = np.array([0, 0, 0, 10, 0, 0, 1, 1, 0, 1, 1, 1])
f, grad = cv.get_cv("distance", [0, 2])
print("\nDistance: ", f)
print("Gradient:\n", grad)

f = cv.angle([0, 1, 2])
print("\nAngle: ", np.degrees(f))
print("Gradient:\n", cv.gradient)

f = cv.torsion([0, 1, 2, 3])
print("\nDihedral: ", np.degrees(f))
print("Gradient:\n", cv.gradient)

f = cv.linear_combination([[1.0, [2, 3]], [2.0, [0, 1]]])
print("\nLinear combination: ", f)
print("Gradient:\n", cv.gradient)

f = cv.distorted_distance([0, [2, 3]])
print("\ndistorted distance: ", f)
print("Gradient:\n", cv.gradient)

d = cv.cec([[0, 1], [[1.0, 2], [2.0, 3]]])
print("\ncec: ", d)
print("Gradient:\n", cv.gradient)

d = cv.gmcec([[0, 1], [[1.0, 2], [1.0, 3]], [[2, 2, 3]]], mapping="default")
print("\ngmcec: ", d)
print("Gradient:\n", cv.gradient)

f, grad = cv.get_cv(
    "lin_comb_custom", [["distance", 1.0, [0, 2]], ["distance", 1.0, [0, 2]]]
)
print("\ncustom linear combination: ", f)
print("Gradient:\n", cv.gradient)
