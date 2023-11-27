# tests/test_lagrangian.py
import numpy as np
import pytest

from src.evolution import lagrangian_eom
from src.potentials import potential, gravity

m = np.random.randint(1, 100)
n = np.random.randint(1, 100)

@pytest.mark.benchmark(min_rounds=100, max_time=5)
def test_lagrangian_eom_one_body_ndim(benchmark):
    mass = np.ones(1)
    x = np.random.random(n)
    x_t = np.random.random(n)

    # call the lagrangian eom
    result = benchmark(lagrangian_eom, x, x_t, mass, [potential(gravity, g=9.81)])

    assert result[1].shape == (n,)    
    assert np.allclose(result[1][:-1], 0.0, atol=1e-5)
    assert np.allclose(result[1][-1], -9.81, atol=1e-5)

@pytest.mark.benchmark(min_rounds=100, max_time=5)
def test_lagrangian_eom_mbodies_ndim(benchmark):
    mass = np.ones(m)
    x = np.random.random((m, n))
    x_t = np.random.random((m, n))

    # call the lagrangian eom
    result = benchmark(lagrangian_eom, x, x_t, mass, [potential(gravity, g=9.81)])

    assert result[1].shape == (m, n)
    assert np.allclose(result[1][:, :-1], 0.0, atol=1e-5)
    assert np.allclose(result[1][:, -1], -9.81, atol=1e-5)