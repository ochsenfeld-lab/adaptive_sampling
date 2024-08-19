import torch
import numpy as np


class GRAPH_CV:
    """Collective Variables based on the representation of molecules as graphs with vertices (atoms) and edges (bonds)

    See: Pietrucci et al., Phys. Ref. Lett. (DOI: 10.1103/PhysRevLett.107.085504)

    Args:
        atom_indices: indices of included atoms
        atom_types: types of included atoms (implemented types: "H", "C", "N", "O")
        N: exponent of nominator of switching function
        M: exponent od denominator of switching function
        parallel: if multiprocessing is used for calculation of adjacency matrix
        requires_grad: if gradient is calculated using `torch.autograd`, not possible if `parallel=True`
    """

    def __init__(
        self,
        atom_indices: list = [],
        atom_types: list = [],
        N: int = 6,
        M: int = None,
        parallel: bool = False,
        requires_grad: bool = True,
    ):
        self.atom_indices = atom_indices
        if not len(self.atom_indices):
            raise ValueError(" >>> GRAPH_CV: Missing atom indices")
        self.atom_types = atom_types
        if not len(self.atom_types) or len(self.atom_types) != len(self.atom_indices):
            raise ValueError(
                " >>> GRAPH_CV: Number of atom types does not match number of atoms"
            )
        self.natoms = len(self.atom_indices)
        self.sqrt_natoms = torch.sqrt(torch.tensor(self.natoms))
        self.exponent_N = N
        self.exponent_M = M
        if self.exponent_M == None:
            self.exponent_M = 2 * self.exponent_N
        self.parallel = parallel

        self.requires_grad = requires_grad
        if self.requires_grad:
            torch.autograd.set_detect_anomaly(True)
            if self.parallel:
                raise ValueError(
                    " >>> WARNING: cannot use multiprocessing if contact matrix `requires_grad`"
                )

        self.A = torch.zeros(size=(self.natoms, self.natoms))
        self.cv = torch.zeros((1))

    def calc(self, z: np.array):
        """Calculates the maximum eigenvalue of adjacency matrix A and its derivative

        see: Parrinello et al., J. Phys. Chem. Lett., (doi: 10.1021/acs.jpclett.1c03993)

        Args:
            z: full cartesian coordinates

        Returns:
            cv: largest eigenvalue of contact matrixp
        """
        if not torch.is_tensor(z):
            z = torch.from_numpy(z)

        if self.requires_grad:
            z.requires_grad = True

        self.A = self.adjacency_matrix(z)

        L = torch.linalg.eigvals(self.A)
        self.cv = torch.max(torch.real(L))

        if self.requires_grad:
            self.gradient = torch.autograd.grad(self.cv, z, allow_unused=True)[0]
            self.gradient = self.gradient.detach().numpy()

        return self.cv

    def adjacency_matrix(self, z):
        """Calculates symmetric adjacency matrix of molecular graph

        Note: can get slow for large number of atoms (N(N-1)/2 scaling)

        Args:
            z: vector of atomic coordinates
            parallel: enable multiprocessing (breaks autograd!)

        Returns:
            A: contact matrix
        """
        z = z.view(int(torch.numel(z) / 3), 3)[self.atom_indices]
        A = torch.zeros_like(self.A)

        if self.parallel:
            import torch.multiprocessing as mp

            with mp.Pool() as pool:
                pool.starmap(self._calc_a_i, [(i, z, A) for i in range(self.natoms)])

        else:
            for i in range(self.natoms):
                self._calc_a_i(i, z, A)

        return A

    def _calc_a_i(self, i: int, z: torch.tensor, A: torch.tensor):
        """Inner loop for calculation of adjacency matrix A

        Args:
            i: index of outher iteration
            z: coordinate vector
            A: adjacency matrix, row and column i is overwritten with new a_ij
        """
        # loop runs over upper triangular matrix
        for j, type_j in enumerate(self.atom_types[i:-1], start=i + 1):
            r_ij = torch.linalg.norm(z[i] - z[j])
            r_0 = GRAPH_CV.get_sigma_ij(self.atom_types[i], type_j)
            diff = r_ij / r_0
            if diff == 1.0:
                # avoid zero division
                diff += 0.00001
            a = (1.0 - torch.pow(diff, self.exponent_N)) / (
                1.0 - torch.pow(diff, self.exponent_M)
            )
            A[i, j] = a
            A[j, i] = a

    @staticmethod
    def get_sigma_ij(type_i, type_j):
        """get equilibrium bond length from atom types

        Note: Values are taken from Zheng et al. (doi: 10.1021/jp500398k) and not further optimized
        """
        bond_type = "".join(sorted(type_i + type_j))
        if bond_type.upper() == "CC":
            return 2.65
        elif bond_type.upper() == "OO":
            return 2.65
        elif bond_type.upper() == "NN":
            return 2.65
        elif bond_type.upper() == "CO":
            return 2.65
        elif bond_type.upper() == "CN":
            return 2.65
        elif bond_type.upper() == "NO":
            return 2.65
        elif bond_type.upper() == "CH":
            return 2.22
        elif bond_type.upper() == "HO":
            return 2.22
        elif bond_type.upper() == "HN":
            return 2.22
        elif bond_type.upper() == "HH":
            return 2.22
        elif bond_type.upper() == "PP":
            return 2.65
        elif bond_type.upper() == "HP":
            return 2.22
        elif bond_type.upper() == "CP":
            return 2.65
        elif bond_type.upper() == "NP":
            return 2.65
        elif bond_type.upper() == "OP":
            return 2.65

        else:
            raise NotImplementedError(f" >>> Unknown bond type {bond_type}")
