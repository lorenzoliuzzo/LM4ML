from trash.surfaces import Sphere
from potentials import elastic
from trash.lagrangian import constrained_lagrangian, constrained_lagrangian_eom, integrate_constrained_lagrangian_eom, integrate_lagrangian_eom

from jax import numpy as jnp
import timeit


# Initial conditions
mass = 1.0
q0 = jnp.array([[0.0, 0.5]]) * jnp.pi
q_t0 = jnp.array([[0.1, 0.0]])

# Creating the surface
surf = Sphere(radius=1.0)

# setting the potential parameters
elastic_pot_params = {'k': 50, 'fixed_pos': jnp.array([0.0, 0.0, 2.0])}

# Creating a potential energy as a list
potential = [(elastic, elastic_pot_params)]

# setting the integration time span
t_span = jnp.linspace(0.0, 10.0, 1000)

print(integrate_lagrangian_eom(q0, q_t0, mass, t_span, potential))

# time = timeit.timeit(lambda: integrate_constrained_lagrangian_eom(surf, q0, q_t0, mass, t_span, potential), number=1000000)
# print(f"JIT version time: {time:.6f} seconds")