# tests/test_gravity.py
import numpy as np
from src.lagrangian import lagrangian, lagrangian_eom
from src.potentials import potential_energy, gravity

def test_gravity():
    # setting a casual number of dimensions and bodies
    ndim = np.random.randint(1, 3)
    nbodies = np.random.randint(1, 100)
    print("ndim", ndim, "nbodies", nbodies)

    # setting the initial conditions
    mass = np.random.randn(nbodies)
    x = np.random.random((nbodies, ndim))
    x_t = np.random.random((nbodies, ndim))

    # create the potential energy
    g_pot = potential_energy(gravity, g=9.81)   

    # call the lagrangian eom
    q_t, q_tt = lagrangian_eom(x, x_t, mass, potentials=[g_pot])
    
    # asserting the results
    rtol = 1.e-6
    assert np.allclose(q_t, x_t, rtol=rtol)
    assert np.allclose(q_tt[:, :-1], 0.0, rtol=rtol)
    assert np.allclose(q_tt[:, -1], -9.81, rtol=rtol)    
