from trash.lagrangian import Lagrangian
from potentials import gravitational
from jax import numpy as jnp
from jax import vmap, grad, hessian, jacfwd

# Initial conditions
mass = jnp.array([5.972e24, 7.342e22])  # Earth and Moon mass in kg
q0 = jnp.array([[0.0, 0.0, 0.0], [384.4e6, 0.0, 0.0]])  # Initial positions in meters
q_t0 = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.1e4, 1.1e4]])

# Creating the lagrangian
L = Lagrangian(potentials=[(gravitational,{})])
print(L(q0, q_t0, mass))
print(gravitational(q0, q_t0, mass))

t_span = jnp.linspace(0.0, 31.0e6, 50000)
L.draw_trajectory(q0, q_t0, mass, t_span)