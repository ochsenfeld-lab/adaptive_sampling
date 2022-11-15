import os
import itertools
import torch
from scipy.spatial import KDTree

from adaptive_sampling.units import BOHR_to_ANGSTROM


def read_xyz(xyz_name: str) -> torch.tensor:
    """Read cartesian coordinates from file (*.xyz)

    Args:
        xyz_name (str): file-name of xyz-file

    Returns:
        mol: (3*natoms,) cartesian coordinates
    """
    if os.path.exists(xyz_name):
        xyzf = open(xyz_name, "r")
    else:
        raise ValueError(" >>> fatal error: xyz file not found")

    mol = []
    for line in xyzf:
        words = line.strip().split()
        if len(words) >= 4:
            mol.append([float(words[1]), float(words[2]), float(words[3])])

    mol = itertools.chain(*mol)
    mol = torch.FloatTensor(list(mol)) / BOHR_to_ANGSTROM
    return mol


def read_traj(xyz_name: str) -> list:
    """Read cartesian coordinates of trajectory from file (*.xyz)

    Args:
        xyz_name (str): file-name of xyz-file

    Returns:
        [[float,float,float],...]: structure as listimport itertools of atom-data, e.g.: [[0.0,0.0,0.0],[0.74,0.0,0.0]]
    """
    if os.path.exists(xyz_name):
        xyzf = open(xyz_name, "r")
    else:
        raise ValueError(" >>> fatal error: xyz file not found")

    traj = []
    mol = []
    n = 0
    for i, line in enumerate(xyzf):
        words = line.strip().split()
        if i == 0:
            n_atoms = int(words[0])
        elif n == n_atoms:
            n = 0
            mol = itertools.chain(*mol)
            mol = torch.FloatTensor(list(mol)) / BOHR_to_ANGSTROM
            traj.append(mol)
            mol = []
        elif len(words) >= 4:
            n += 1
            mol.append([float(words[1]), float(words[2]), float(words[3])])

    return traj


def interpolate_coordinates(images: list, n_interpol: int = 20) -> list:
    """linear interpolation between cartesian coordinates

    Args:
        images: list of cartesian coordinates of images
        n_interpol: number of images that are created between two nodes of initial path

    Returns:
        path: cartesian coordinates of original and interpolated images
    """
    path = []
    for i in range(len(images) - 1):
        a = images[i]
        b = images[i + 1]
        r = b - a
        d = torch.linalg.norm(r)
        unit_vec = r / d
        step = d / n_interpol

        path.append(a)
        for j in range(1, n_interpol):
            # linear interpolation
            p = a + unit_vec * j * step
            path.append(p)

    path.append(b)
    return path


def find_closest_points(coords: torch.tensor, indices: list = None) -> list:

    coords_tree = coords.detach()
    coords_tree = torch.reshape(coords_tree, (int(len(coords_tree) / 3), 3)).float()

    if indices != None:
        coords_tree = coords_tree[indices]

    kdtree = KDTree(coords_tree)

    close_indices = []
    for xyz in coords_tree:
        # find three closest atoms (first point is same atom)
        _, indices = kdtree.query(xyz, 4)
        close_indices.append(indices[1:])

    return close_indices


def get_internal_coords(
    coords: torch.tensor, index_pairs: list, #indices: list = None
) -> torch.tensor:
    coords = torch.reshape(coords, (int(len(coords) / 3), 3)).float()
    internals = []
    for pair in index_pairs:
        dist = torch.linalg.norm(coords[pair[1]] - coords[pair[0]])
        internals.append(dist)
    internals = torch.stack(internals)
    return internals


def rmsd(V, W):
    """root-mean-square deviation"""
    diff = V - W
    return torch.sqrt(torch.sum(diff * diff) / len(V))


def kabsch_rmsd(
    coords1: torch.tensor,
    coords2: torch.tensor,
    indices: bool = None,
) -> torch.tensor:
    """minimize rmsd between cartesian coordinates by kabsch algorithm

    Args:
        coords1: (3*n_atoms,) tensor of cartesian coordinates
        coords2: (3*n_atoms,) tensor of cartesian coordinates
        indices: indices of atoms that are included

    Returns:
        coords1: (3*n_atoms,) coordinates fitted to coords2
    """
    n = len(coords1)
    coords1 = torch.reshape(coords1, (int(n / 3), 3)).float()
    coords2 = torch.reshape(coords2, (int(n / 3), 3)).float()

    if indices != None:
        coords1 = coords1[indices]
        coords2 = coords2[indices]

    # translate centroids of molecules onto each other
    coords1_new = coords1 - centroid(coords1)
    coords2_new = coords2 - centroid(coords2)

    # optimal rotation of coords1 to fit coords2
    coords1_new = kabsch_rot(coords1_new, coords2_new)
    return rmsd(coords1_new, coords2_new)


def centroid(X: torch.tensor) -> torch.tensor:
    """Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    Args:
        X: (Natoms, 3) array of cartesian coordinates

    Returns:
        C: 3D centroid
    """
    C = X.mean(axis=0)
    return C


def kabsch_rot(coords1: torch.tensor, coords2: torch.tensor) -> torch.tensor:
    """Rotate coords1 on coords2

    Args:
        coords1: (3*n_atoms,) tensor of cartesian coordinates
        coords2: (3*n_atoms,) tensor of cartesian coordinates

    Returns:
        new_coords1: (3*n_atoms,) tensor of rotated coordinates
    """
    U = kabsch(coords1, coords2)
    coords1_new = torch.matmul(coords1, U)
    return coords1_new


def kabsch(P: torch.tensor, Q: torch.tensor):
    """Kabsch algorithm to obtain rotation matrix"""
    C = torch.matmul(torch.transpose(P, 0, 1), Q)
    V, S, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = torch.matmul(V, W)
    return U


def quaternion_rmsd(
    coords1: torch.tensor,
    coords2: torch.tensor,
    indices: list = None,
    reshape: bool = True,
    return_coords: bool = False,
) -> torch.tensor:
    """
    Rotate coords1 on coords2 and calculate the RMSD
    based on doi:10.1016/1049-9660(91)90036-O

    Args:
        coords1 : (natoms*3,) tensor of cartesian coordinates
        coords3 : (natoms*3,) tensor of cartesian coordinates
        indices: list of indices that are included
        reshape: if False, coords1 not reshaped to (natoms,3)
        return_coords: if transformed coords1 are returned

    Returns:
        rmsd: minimized root-mean-square deviation
        coords1: if return_coords
    """
    if reshape:
        coords1 = torch.reshape(coords1, (int(len(coords1) / 3), 3)).float()
        if indices != None:
            coords1 = coords1[indices]

    coords2 = torch.reshape(coords2, (int(len(coords2) / 3), 3)).float()
    if indices != None:
        coords2 = coords2[indices]

    # translate centroids of molecules onto each other
    coords1_new = coords1 - centroid(coords1)
    coords2_new = coords2 - centroid(coords2)

    rot = quaternion_rotate(coords1_new, coords2_new)
    coords1_new = torch.matmul(coords1_new, rot)

    if return_coords:
        return rmsd(coords1_new, coords2_new), coords1_new
    return rmsd(coords1_new, coords2_new)


def _quaternion_transform(r: torch.tensor) -> torch.tensor:
    """Get optimal rotation"""
    Wt_r = _makeW(*r).T
    Q_r = _makeQ(*r)
    rot = torch.matmul(Wt_r, Q_r)[:3, :3]
    return rot


def _makeW(r1, r2, r3, r4=0):
    """matrix involved in quaternion rotation"""
    W = torch.tensor(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return W


def _makeQ(r1, r2, r3, r4=0):
    """matrix involved in quaternion rotation"""
    Q = torch.tensor(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return Q


def quaternion_rotate(X: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """
    Calculate the rotation

    Args:
        X: (natoms,3) tensor of cartesian coordinates
        Y: (natoms,3) tensor of cartesian coordinates

    Returns:
        rot: Rotation matrix (3,3)
    """
    N = X.size(dim=0)
    W = torch.stack([_makeW(*Y[k]) for k in range(N)])
    Q = torch.stack([_makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = torch.stack([torch.matmul(Q[k].T, W[k]) for k in range(N)])
    A = torch.sum(Qt_dot_W, axis=0)
    eigen = torch.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = _quaternion_transform(r)
    return rot


def get_amber_charges(prmtop: str) -> list:
    """Parse charges from AMBER parameter file 

    Args:
        prmtop (string): filename

    Returns:
        charges (list): atom charges in a.u.
    """
    with open(prmtop, "r") as f:
        prm = f.readlines()
        for i, line in enumerate(prm):
            if line.find("CHARGE") != -1:
                q_str = prm[i+2]
                j = 3
                while len(prm) > j:
                    if prm[i+j].find("FLAG") != -1:
                        break
                    else:
                        q_str += prm[i+j]
                        j += 1
                break

    charge = []
    for q in q_str.split(" "):
        if q:
            charge.append(float(q) / 18.2223)  # converted to a.u. with factor sqrt(electrostatic constant)

    return charge