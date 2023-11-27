# tests/test_lagrangian.py
import numpy as np
import pytest

from src.lagrangian import lagrangian
from src.potentials import potential, gravity

m = np.random.randint(1, 100)
n = np.random.randint(1, 100)

def test_lagrangian_one_body_ndim():
    mass = np.ones(1)
    x = np.random.random(n)
    x_t = np.random.random(n)

    # call the lagrangian
    L = lagrangian(x, x_t, mass, [potential(gravity, g=9.81)])
    assert L.shape == (1,)
    
def test_lagrangian_m_bodies_ndim():
    mass = np.ones(m)
    x = np.random.random((m, n))
    x_t = np.random.random((m, n))

    # call the lagrangian
    L = lagrangian(x, x_t, mass, [potential(gravity, g=9.81)])
    assert L.shape == (m,)