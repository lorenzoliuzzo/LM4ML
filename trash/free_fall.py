import jax
from jax import numpy as jnp
from lagrangian import lagrangian, evolve_lagrangian, draw_trajectory
from potentials import gravity
from matplotlib import pyplot as plt

# Example usage
nbodies = 2
ndim = 3

# setting the ic
m1 = 1.0
x0 = [1.0, 0.0, 5.0]
x1 = [0.0, 1.0, 10.0]
x_t0 = [-0.2, 0.0, 2.0]
x_t1 = [1.6, 0.0, 0.0]

# Convert to JAX arrays
q = jnp.array([x0, x1])
q_t = jnp.array([x_t0, x_t1])
mass = jnp.array([m1, m1])

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)

# Create the gravity potential with its parameters
g_pot = (gravity, {'g': 9.81})

# call the lagrangian function
L, eom = lagrangian(q, q_t, mass, potentials=[g_pot])
print(L)
print(eom)

# evolving the lagrangian
positions, _ = evolve_lagrangian(q, q_t, mass, t_span=jnp.linspace(0., 50., 200), potentials=[g_pot])

# plotting the trajectory
draw_trajectory(positions)