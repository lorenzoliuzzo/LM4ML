import jax
from jax import numpy as jnp
from lagrangian import Lagrangian
from potentials import gravity

# Example usage
nbodies = 2
ndim = 3

# setting the ic
m1 = 1.0
m2 = 2.0
x0 = 3 * jnp.ones(3)
x1 = jnp.array([2.0, -1.0, 5.0])
x_t0 = jnp.ones(3)
x_t1 = jnp.array([0.0, 0.0, 1.0])

# Convert to JAX arrays
q = jnp.array([x0, x1])
q_t = jnp.array([x_t0, x_t1])
mass = jnp.array([m1, m2])

print("q", q)
print("q_t", q_t)
print("mass", mass)

# Create the lagrangian
L = Lagrangian(potentials=[(gravity, {'g': 9.81})])
L_v = L(q, q_t, mass)
print(L_v)
print(L.eom(q, q_t, mass))

L.draw_trajectory(q, q_t, mass, jnp.linspace(0., 2., 100))