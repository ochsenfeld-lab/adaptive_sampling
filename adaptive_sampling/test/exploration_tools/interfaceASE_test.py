import pytest
import numpy as np
from adaptive_sampling.interface.interfaceASE import AseMD

test_ase = AseMD()

bo_acetaldehyd_round = np.array([[0.0,2.0,1.0,1.0],[2.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]])
print(bo_acetaldehyd_round)
