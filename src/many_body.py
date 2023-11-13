import jax
from jax import numpy as jnp
from lagrangian import constrained_lagrangian, evolve_lagrangian, draw_trajectory
from potentials import gravity, elastic
from surfaces import sphere
from matplotlib import pyplot as plt

import numpy as np

# Example usage
nbodies = 5
ndim = 2

# setting the ic
q = np.random.rand(nbodies, ndim)
q_t = np.random.rand(nbodies, ndim)
mass = np.random.rand(nbodies)

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)

# Create the surface with its parameters
sphere_params = {'radius': 30.0, 'center': jnp.ones(3)}
surf = (sphere, sphere_params)

# Create the potential with its parameters
g_pot = (gravity, {'g': 9.81})

# Create the lagrangian
L, eom = constrained_lagrangian(surf, q, q_t, mass, potentials=[g_pot])
print("L", L)
print("eom", eom)
print("shapes:", eom[0].shape, eom[1].shape)

tmax = 5.
npoints = 500
positions, _ = evolve_lagrangian(q, q_t, mass, t_span=jnp.linspace(0., tmax, npoints), potentials=[], surface=surf)

draw_trajectory(positions, surface=surf)