import pytest
import numpy as np

from adaptive_sampling.sampling_tools.feneb import FENEB

def test_init_path():
    feneb = FENEB(1.0, 10.0, ["resources/feneb_start.xyz", "resources/feneb_end.xyz"], nimages=1)
    result = [
        np.asarray([[0.0, 0.0, 0.0]]), 
        np.asarray([[0.5,0.0,0.0]]), 
        np.asarray([[1.0,0.0,0.0]]),
    ]
    assert np.allclose(feneb.path, result, atol=1.0e-5)

def test_tangent_vector():
    feneb = FENEB(1.0, 10.0, ["resources/feneb_start.xyz", "resources/feneb_end.xyz"], nimages=1)
    tau = feneb.tangent_vector([feneb.path[0], feneb.path[1], feneb.path[2]])
    result = [
        np.asarray([[1.0, 0.0, 0.0]]), 
    ]
    assert np.allclose(tau, result, atol=1.0e-5)

def test_opt_node_spacing():
    feneb = FENEB(
        100.0, 
        100.0, 
        ["resources/feneb_start.xyz", "resources/feneb_end.xyz"], 
        nimages=1, 
        maxiter_spring=10000, 
        conf_spring=1.0e-3,
    )
    feneb.path[1] = np.asarray([[0.4,0.0,0.0]])
    feneb.opt_node_spacing()
    result = [
        np.asarray([[0.0, 0.0, 0.0]]), 
        np.asarray([[0.5,0.0,0.0]]), 
        np.asarray([[1.0,0.0,0.0]]),
    ]
    assert np.allclose(feneb.path, result, atol=1.0e-3)

