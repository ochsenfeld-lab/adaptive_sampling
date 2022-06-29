import os
import itertools
import torch

def read_xyz(xyz_name: str) -> torch.tensor:
    '''Read cartesian coordinates from file (*.xyz)

    Args:
        xyz_name (str): file-name of xyz-file

    Returns:
        mol: (3*natoms,) cartesian coordinates
    '''
    if os.path.exists(xyz_name):
        xyzf = open(xyz_name,'r')
    else:
        raise ValueError(" >>> fatal error: xyz file not found")   

    mol = []
    for line in xyzf:
        words = line.strip().split()
        if len(words) >= 4:
            mol.append([float(words[1]),float(words[2]),float(words[3])])

    mol = itertools.chain(*mol)
    mol = torch.FloatTensor(list(mol))
    return mol

def read_traj(xyz_name: str) -> list:
    '''Read cartesian coordinates of trajectory from file (*.xyz)

    Args:
        xyz_name (str): file-name of xyz-file

    Returns:
        [[float,float,float],...]: structure as listimport itertools of atom-data, e.g.: [[0.0,0.0,0.0],[0.74,0.0,0.0]]
    '''
    if os.path.exists(xyz_name):
        xyzf = open(xyz_name,'r')
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
            mol = torch.FloatTensor(list(mol))
            traj.append(mol)
            mol = []
        elif len(words) >= 4:
            n += 1
            mol.append([float(words[1]),float(words[2]),float(words[3])])

    return traj

def interpolate_coordinates(images: list, n_interpol: int=20):
    """linear interpolation between cartesian coordinates
    
    Args:
        images: list of cartesian coordinates of images
        n_interpol: number of images that are created between two nodes of initial path

    Returns:
        path: cartesian coordinates of original and interpolated images
    """
    path = []
    for i in range(len(images)-1):
        a = images[i]
        b = images[i+1]
        r = b - a
        d = torch.linalg.norm(r)
        unit_vec = r / d
        step = d / n_interpol
                
        # create interploated images
        path.append(a)
        for _ in range(n_interpol):
            a += unit_vec * step 
            path.append(a)

    path.append(b)
    return path

def rmsd(V, W):
    """root-mean-square deviation"""
    diff = V - W
    return torch.sqrt(torch.sum(diff * diff) / len(V))

def kabsch_rmsd(coords1: torch.tensor, coords2: torch.tensor) -> torch.tensor:
    """minimize rmsd between cartesian coordinates by kabsch algorithm
    
    Args:
        coords1: (3*n_atoms,) tensor of cartesian coordinates
        coords2: (3*n_atoms,) tensor of cartesian coordinates

    Returns:
        coords1: (3*n_atoms,) coordinates fitted to coords2
    """
    n = len(coords1)
    coords1 = torch.reshape(coords1, (int(n/3), 3)).float()
    coords2 = torch.reshape(coords2, (int(n/3), 3)).float()

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

def kabsch(P, Q):
    """Kabsch algorithm to obtain rotation matrix"""
    C = torch.matmul(torch.transpose(P, 0, 1), Q)
    V, S, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    
    U = torch.matmul(V, W)
    return U
