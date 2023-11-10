from lagrangian import Lagrangian
from potentials import gravitational, elastic
from jax import numpy as jnp
from jax import grad, hessian

# Initial conditions
mass = 1
q0 = jnp.array([0.0, 0.0, 400.0])
q_t0 = jnp.array([30.0, 0.0, 0.0])

# setting the potential parameters
gravitational_pot_params = {'G': 9.81}
elastic_pot_params = {'k': 50, 'fixed_pos': jnp.zeros(3)}

# Creating the lagrangian
L = Lagrangian(potentials=[(gravitational, gravitational_pot_params), (elastic, elastic_pot_params)])
print(L(q0, q_t0, mass))

# Define the time span for integration
t_span = jnp.linspace(0.0, 10.0, 100)
q, q_t = L.integrate_eom(q0, q_t0, mass, t_span)