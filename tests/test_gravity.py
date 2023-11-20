# tests/test_gravity.py
import numpy as np
from src.lagrangian import lagrangian, lagrangian_eom
from src.potentials import potential_energy, gravity

def test_gravity():
    # setting the initial conditions
    ndim = 3
    nbodies = np.random.randint(1, 10)
    mass = np.random.randn(nbodies)
    x = np.random.random((nbodies, ndim))
    x_t = np.random.random((nbodies, ndim))

    # create the potential energy
    g_pot = potential_energy(gravity, g=9.81)   

    # call the lagrangian function
    L = lagrangian(x, x_t, mass, potentials=[g_pot])
    q_t, q_tt = lagrangian_eom(x, x_t, mass, potentials=[g_pot])
    
    # asserting the results from the equation of motion
    assert np.allclose(q_t, x_t)
    assert np.allclose(q_tt[:, -1], -9.81)