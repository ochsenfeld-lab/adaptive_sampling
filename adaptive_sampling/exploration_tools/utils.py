from ase import Atoms
import numpy as np
from ..units import *    

def fib(n):
    """
    :param n: Fibonacci number to be calculated (index in the series)
    :return: Fibonacci number

    The function fib returns a Fibonacci number without calculating the entire series.
    The round function at the end reduces the obtained result to an int.
    """
    # Calculate golden ratio
    phi = (1 + (5 ** (1 / 2))) / 2

    # Binet's function to calculate Fibonacci number of a given index n
    return round((phi ** n - ((1 - (5 ** (1 / 2))) / 2) ** n) / (5 ** (1 / 2)))

def fibonacci_sphere(samples = 1, dr = 1, inc = 0):
    """
    :param samples: how many points shall be generated on a fibonacci sphere
    :param dr: allows to choose a radius different than 1
    :param inc: increment for rotating the different subshells relative to eachother
    :return: points on a fibonnaci sphere in cartesian coordinates

    The function initializes points on the surface of a fibonacci sphere acoording to the
    golden angle. This ensures an optimal spacing between points.
    """
    points = []
    golden_ang = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    radius = 1 # unit sphere
    inc_rad = inc/180*np.pi

    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2
        theta = np.arccos(z) + inc_rad
        phi = golden_ang * i

        x = dr * radius * np.sin(theta) * np.cos(phi)
        y = dr * radius * np.sin(theta) * np.sin(phi)
        z = dr * radius * np.cos(theta)

        points.append((x, y, z))

    return points

def rotate_random(atoms: Atoms):
    atoms.rotate(a=360 * np.random.rand(), v='x', center='COM')
    atoms.rotate(a=360 * np.random.rand(), v='y', center='COM')
    atoms.rotate(a=360 * np.random.rand(), v='z', center='COM')

    return atoms

def spheric_fib_init(mols, nmolecules, dr):

    # Initialize xyz object with dummy H
    temp_mol = Atoms('H', positions=[(0, 0, 0)]) 

    # Create array with nmolecules molecules of each sort
    molsArray = []

    for i_mol, init_mol in enumerate(mols):
        #print(init_mol)
        for i in range(nmolecules[i_mol]):
            molsArray.append(init_mol.copy())

    # Shuffle elements in the molsArray
    np.random.shuffle(molsArray)

    # Start counting how many molecules have been already placed
    placedMolecules = 0

    # Initialize variable for Fibonacci sum
    sumFib = 0

    # Initialize difference between Fibonacci sum and total number of molecules to be placed
    diff = sumFib - sum(nmolecules)

    # Start with 3 molecules in the most inner shell
    offset = 4

    # Initialize distance between shells
    drad = 0

    # As long as total Fibonacci sum < (number of total molecules) place molecules:
    for n in range(offset,30,1):
        if (n % 2) == 0:
            multi = n - offset
        else:
            multi = n - 1 - offset

        inc = multi * 5 / 2 + (n % 2) * 180 # degrees

        # Calculate number of molecules to be placed in the shell with the Binet's formula
        samples = fib(n)
        drad += dr

        #print("Radius = %14.7f Angstrom" %(drad))
        # Generate grid of points for each shell
        points = fibonacci_sphere(samples, drad, inc)

        # Place molecules on the subsphere (shell)
        for i in range(len(points)):
            if placedMolecules == len(molsArray):  # stop when all molecules have been placed
                break
            add_mol = rotate_random(molsArray[placedMolecules])
            translation = points[i]
            add_mol.translate(translation)
            temp_mol = temp_mol + add_mol

            placedMolecules = placedMolecules + 1

        # Update sum, dr and diff and iterate the index in the Fibonacci series to generate the next shell (subsphere)
        sumFib += samples
        diff = sumFib - sum(nmolecules)

        if diff >= 0:
            break

    # Delete dummy atom
    del temp_mol[0]

    mol = temp_mol

    return mol, drad

def write_compressed_bo(bo_list: list, output_file: str = "bond_order") -> None:
    """Compress bond order file to npz file.

    Args:
        bond_list: list of sparse, upper triangular, indexed bond orders for a corresponding time step (index + 1) * out_freq
        output_file: output file to be saved
    """
    np.savez_compressed(f"{output_file}.npz", *bo_list)

def bo_matrix_from_npz(natoms: int, npz_file: str = "bond_order.npz") -> np.array:
    """Read bond order matrix from npz file.

    Args:
        npz_file: path to npz file
    Returns:
        bo_matrices_list: list of bond order matrices
    """
    loaded_data = np.load(npz_file)
    bond_order_indexed = [loaded_data[key] for key in loaded_data]
    timesteps = len(bond_order_indexed)
    bo_matrices_list = []
    for t in range(timesteps):
        bond_order_matrix = np.zeros((natoms, natoms))
        for bo in bond_order_indexed[t]:
            bond_order_matrix[int(bo[1]), int(bo[2])] = bo[0]
            bond_order_matrix[int(bo[2]), int(bo[1])] = bo[0]
        bo_matrices_list.append(bond_order_matrix)
    return bo_matrices_list

def triu_bo_sparse(bo: np.array, zero_tresh: float = 0.5) -> np.array:
    """Reduce sparse bond order matrix to indexed non-zero bond orders in upper triangular matrix.

    Args:
        bo: sparse bond order matrix
        zero_tresh: threshold, under which bond order matrix entries are rounded to zero
    Returns:
        bo_indexed: array of form ([bo,i,j],...) with non-zero bond orders and corresponding indices in bo-matrix
    """
    bo_rounded = np.copy(bo)
    bo_rounded[bo < zero_tresh] = 0.0
    natoms = bo.shape[0]
    bo_indexed = []
    for i in range(0,natoms):
        for j in range(i+1,natoms):
            if bo_rounded[i,j] != 0.0:
                bo_indexed.append([float(bo_rounded[i][j]), int(i), int(j)])
    return np.array(bo_indexed)
