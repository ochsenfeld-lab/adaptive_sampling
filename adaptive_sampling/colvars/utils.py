import os
import itertools
import torch
from adaptive_sampling.units import BOHR_to_ANGSTROM


def read_xyz(
    xyz_name: str
) -> torch.tensor:
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

def read_path(
    filename: str, 
    ndim: int,
) -> tuple:
    """Read coordinates of path nodes from file (*.xyz, *.npy)
        
    Args:
        filename 
        
    Returns:
        traj: list of torch arrays containing coordinates of nodes
        nnodes: number of nodes in path
    """
    if filename[-3:] == "dcd":
        # TODO: Read path from dcd file
        raise NotImplementedError(" >>> ERROR: Reading path from dcd not yet implemented. Use `.xzy` or `.npy` file")

    elif filename[-3:] == "npy":
        import numpy
        traj = numpy.load(filename)
        traj = [torch.from_numpy(t) for t in traj]

    elif filename[-3:] == "xyz":
        with open(filename, "r") as xyzf:
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
                    if ndim == 2:
                        mol.append([float(words[1]), float(words[2])])
                    else:
                        mol.append([
                            float(words[1]) / BOHR_to_ANGSTROM, 
                            float(words[2]) / BOHR_to_ANGSTROM, 
                            float(words[3]) / BOHR_to_ANGSTROM,
                        ])
                    n += 1

        if mol:
            mol = itertools.chain(*mol)
            mol = torch.FloatTensor(list(mol))
            traj.append(mol)
    
    else:
        raise ValueError(" >>> ERROR: Unnown format of path. Available: `.xzy`, `.npy`, `.dcd`.")

    return traj, len(traj)


def get_rmsd(V: torch.tensor, W: torch.tensor):
    """root-mean-square deviation"""
    diff = V - W
    return torch.sqrt(torch.sum(diff * diff) / len(V))


def get_msd(V: torch.tensor, W: torch.tensor):
    """mean-square deviation"""
    diff = V - W
    return torch.sum(diff * diff) / len(V)


def kabsch_rmsd(
    coords1: torch.tensor,
    coords2: torch.tensor,
    indices: bool = None,
    return_coords: bool=False,
    ndim: int=3,
) -> torch.tensor:
    """minimize rmsd between cartesian coordinates by kabsch algorithm

    Args:
        coords1: (3*n_atoms,) tensor of cartesian coordinates
        coords2: (3*n_atoms,) tensor of cartesian coordinates
        indices: indices of atoms that are included
        return_coords: if only transformed coords should be returned
    Returns:
        rmsd: root-mean-squared deviation after fit
    """
    coords1 = coords1.view(int(torch.numel(coords1) / ndim), ndim).float()
    coords2 = coords2.view(int(torch.numel(coords2) / ndim), ndim).float()

    if indices != None:
        coords1 = coords1[indices]
        coords2 = coords2[indices]

    # translate centroids of molecules onto each other
    coords1_new = coords1 - centroid(coords1)
    coords2_new = coords2 - centroid(coords2)

    # optimal rotation of coords1 to fit coords2
    coords1_new = kabsch_rot(coords1_new, coords2_new)

    if return_coords:
        return coords1_new, coords2_new
    return get_rmsd(coords1_new, coords2_new)


def centroid(
    X: torch.tensor,
) -> torch.tensor:
    """Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    Args:
        X: (Natoms, 3) array of cartesian coordinates

    Returns:
        C: 3D centroid
    """
    C = X.mean(axis=0)
    return C


def kabsch_rot(
    coords1: torch.tensor, 
    coords2: torch.tensor,
) -> torch.tensor:
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
        return get_rmsd(coords1_new, coords2_new), coords1_new
    return get_rmsd(coords1_new, coords2_new)


def _quaternion_transform(
    r: torch.tensor,
) -> torch.tensor:
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


def quaternion_rotate(
    X: torch.tensor, 
    Y: torch.tensor,
) -> torch.tensor:
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


def get_internal_coordinate(
    cv: list, 
    coords: torch.tensor, 
    ndim: int=3,
) -> torch.tensor:
    """Get internal coordinate 

    Args:
        cv: definition of internal coordinate
            Available:
                ["distance",     [idx0, idx1]]
                ["angle",        [idx0, idx1, idx2]]
                ["torsion",     [idx0, idx1, idx2, idx3]]
                ["min_distance", [[idx0, idx1], [idx2, idx3], ...]
                ["coordination_number", [[idx0, idx1], [...], r_0, exp_nom, exp_denom]]]
        coords: Cartesian coordinates
        ndim: Number of dimensions of coords
    
    Returns:
        cv: internal coordinate
    """
    z = coords.view(int(torch.numel(coords)/ndim), ndim)
    
    if cv[0].lower() == "distance":
        xi = torch.linalg.norm(z[cv[1][0]] - z[cv[1][1]])

    elif cv[0].lower() == "angle":
        q12 = z[cv[1][0]] - z[cv[1][1]]
        q23 = z[cv[1][1]] - z[cv[1][2]]

        q12_n = torch.linalg.norm(q12)
        q23_n = torch.linalg.norm(q23)
                
        q12_u = q12 / q12_n
        q23_u = q23 / q23_n

        xi = torch.arccos(torch.dot(-q12_u, q23_u))  
                
    elif cv[0].lower() == "torsion":
        q12 = z[cv[1][1]] - z[cv[1][0]]
        q23 = z[cv[1][2]] - z[cv[1][1]]
        q34 = z[cv[1][3]] - z[cv[1][2]]

        q23_u = q23 / torch.linalg.norm(q23)

        n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
        n2 = q34 - torch.dot(q34, q23_u) * q23_u

        xi = torch.atan2(
            torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2)
        ) 

    elif cv[0].lower() == "min_distance":
        # returns minimum distance of list of distances
        dists = []
        for x in cv[1]:
            dists.append(torch.linalg.norm(z[x[0]] - z[x[1]]))
        xi = min(dists)
    
    elif cv[0].lower() == "coordination_number":
        
        cv_def    = cv[1]
        exp_denom = int(cv_def[-1])
        exp_nom   = int(cv_def[-2])
        r_0       = float(cv_def[-3]) / BOHR_to_ANGSTROM

        xi = 0.0
        for atoms in cv_def[:-3]:
            r12 = torch.linalg.norm(z[atoms[1]] - z[atoms[0]])

            # for numerical stability
            if abs(r12-r_0) < 1.e-6:
                r = r12 / (r_0 * 1.000001)
            else:
                r = r12 / r_0

            nom   = 1. - torch.pow(r, exp_nom)
            denom = 1. - torch.pow(r, exp_denom)     
            xi += nom / denom

    elif cv[0].lower() == "cec":
        # Modified Center-of-Excess Charge (mCEC) for Proton Transfer (PT)
        from .proton_transfer import PT
        pt_def = cv[1]
        pt_cv = PT(
            r_sw   = pt_def.get("r_sw", 1.4),
            d_sw   = pt_def.get("d_sw", 0.05),
            n_pair = pt_def.get("n_pair", 15), 
            requires_grad = True,
        )
        xi = pt_cv.gmcec(
            coords, 
            pt_def["proton_idx"],
            pt_def["heavy_idx"],
            pt_def["heavy_weights"],
            pt_def["ref_idx"],
            pair_def=pt_def.get("pair_def", []),
            mapping=pt_def.get("mapping", "default"),
        )

    else:
        raise ValueError(" >>> ERROR: wrong definition of internal coordinate!")
        
    return xi


def cartesians_to_internals(
    coords: torch.tensor, 
    ndim: int=3,
) -> torch.tensor:
    """Converts reduced cartesian coordinates to Z-Matrix

    Args:  
        coords: reduced cartesian coordinates
        ndim: Number of dimensions of input coordinates
        
    Returns:
        zmatrix: Z-Matrix with angles given in radians
    """
        
    z = coords.view(int(torch.numel(coords) / ndim), ndim)
    zmatrix = torch.zeros_like(z)
            
    for i, _ in enumerate(z[1:], start=1):

        zmatrix[i, 0] = get_internal_coordinate(
            ["distance", [i-1, i]], coords, ndim=ndim
        )  

        if i > 1:
            zmatrix[i, 1] = get_internal_coordinate(
                ["angle", [i-2, i-1, i]], coords, ndim=ndim
            ) 
                
        if i > 2:
            zmatrix[i, 2] = get_internal_coordinate(
                ["torsion", [i-3, i-2, i-1, i]], coords, ndim=ndim
            )
    
    return zmatrix


def convert_coordinate_system(
    coords: torch.tensor,
    active: list=None,
    coord_system: str="Cartesian", 
    ndim: int=3,
) -> torch.tensor:
    """Convert XYZ tensor to selected coordinate system

    Args:
        coords: xyz coordinates 
        active: list of active atoms or list of internal coordinates
        coord_system: Selected new coordinate system:
            `Cartesian`: xyz coordinates reduced to `active` atoms
            `ZMatric`: Z-Matrix of `active` atoms
            `CV_space`: Selected internal coordinates (distances, angles and torsions)
        ndim: Number of dimensions of input coordinates

    Returns:
        new_coords: Converted coords 
    """
    if coord_system.lower() == "cartesian":
        z = coords.view(int(torch.numel(coords)/ndim), ndim)
        if active != None:
            z = z[active]
        return z

    elif coord_system.lower() == "zmatrix":
        z = coords.view(int(torch.numel(coords)/ndim), ndim)
        if active != None:
            z = z[active]
        return cartesians_to_internals(z, ndim=ndim)

    elif coord_system.lower() == "cv_space":
        cvs = torch.zeros(len(active))
        for i, cv in enumerate(active):
            cvs[i] = get_internal_coordinate(cv, coords, ndim=ndim)
        return cvs
 