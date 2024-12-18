import pytest
import numpy as np
from ase import Atoms
from adaptive_sampling.interface.interfaceASE import AseMD
from adaptive_sampling.exploration_tools.utils import triu_bo_sparse, write_compressed_bo, bo_matrix_from_npz

test_atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
test_ase = AseMD(atoms=test_atoms)

# ------------------------------------------------------------------------------------------------------------------------------------------
### Test for compressing and decompressing bond order matrices ###
# ------------------------------------------------------------------------------------------------------------------------------------------
# Test data
bo_acetaldehyd_round = np.array([[0.0,2.0,1.0,1.0],[2.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]])
bo_acetaldehyd_triu_sparse = np.array([[2.0,0,1],[1.0,0,2],[1.0,0,3]])
bo_uneven_example = np.array([[0.0,1.2,0.4],[1.2,0.0,1.7],[0.4,1.7,0.0]])
bo_uneven_example_triu_sparse = np.array([[1.2,0,1],[1.7,1,2]])
bo_uneven_example_triu_sparse_recon = np.array([[0.0,1.2,0.0],[1.2,0.0,1.7],[0.0,1.7,0.0]])

def test_triu_bo_sparse():
    assert triu_bo_sparse(bo_acetaldehyd_round) == pytest.approx(bo_acetaldehyd_triu_sparse)

def test_write_compressed_bo():
    output_file = "bond_order"
    write_compressed_bo([bo_acetaldehyd_triu_sparse], output_file)
    assert np.load(f"{output_file}.npz")["arr_0"] == pytest.approx(bo_acetaldehyd_triu_sparse)

def test_bo_matrix_from_npz():
    output_file = "bond_order"
    write_compressed_bo([bo_acetaldehyd_triu_sparse], output_file)
    print(bo_matrix_from_npz(4)[0])
    assert bo_matrix_from_npz(4)[0] == pytest.approx([bo_acetaldehyd_round][0])

def test_bo_compress_decompress():
    output_file = "bond_order"
    bo_list = []
    bo_sparse_ind = triu_bo_sparse(bo_acetaldehyd_round)
    bo_list.append(bo_sparse_ind)
    write_compressed_bo(bo_list, output_file)
    bo_list_recon = bo_matrix_from_npz(4)
    assert bo_list_recon[0] == pytest.approx([bo_acetaldehyd_round][0])

def test_bo_triu_sparse_shorten():
    assert triu_bo_sparse(bo_uneven_example) == pytest.approx(bo_uneven_example_triu_sparse)

def test_bo_compress_decompress_shorten():
    output_file = "bond_order"
    bo_list = []
    bo_sparse_ind = triu_bo_sparse(bo_uneven_example)
    bo_list.append(bo_sparse_ind)
    write_compressed_bo(bo_list, output_file)
    bo_list_recon = bo_matrix_from_npz(3)
    assert bo_list_recon[0] == pytest.approx([bo_uneven_example_triu_sparse_recon][0])
# ------------------------------------------------------------------------------------------------------------------------------------------