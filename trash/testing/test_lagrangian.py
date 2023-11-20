from trash.lagrangian import Lagrangian
from potentials import gravitational, elastic
from jax import numpy as jnp

# Initial conditions
mass = 1.0
q0 = jnp.array([0.0, 0.0, 10.0])
q_t0 = jnp.array([1.0, 0.0, -1.0])

q = jnp.array([q0])
q_t = jnp.array([q_t0])

# setting the potential parameters
# elastic_pot_params = {'k': 5, 'fixed_pos': jnp.zeros(3)}

# Creating the lagrangian 
L = Lagrangian()
print(L(q, q_t, mass))

# Define the time span for integration
t_span = jnp.linspace(0.0, 10.0, 200)
L.draw_trajectory(q, q_t, mass, t_span)