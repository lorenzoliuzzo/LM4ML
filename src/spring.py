import jax
from jax import numpy as jnp
from lagrangian import lagrangian, evolve_lagrangian, draw_trajectory_2d
from potentials import elastic

# setting the ic
m = 1.0
x0 = [1.0, 0.0]
x_t0 = [-0.2, 1.0]

# Convert to JAX arrays
q = jnp.array([x0])
q_t = jnp.array([x_t0])
mass = jnp.array([m])

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)

# Create the gravity potential with its parameters
k_pot = (elastic, {'k': 30, 'fixed_pt': jnp.zeros(2)})

# call the lagrangian function
L, eom = lagrangian(q, q_t, mass, potentials=[k_pot])
print(L)
print(eom)

# evolving the lagrangian
positions, _ = evolve_lagrangian(q, q_t, mass, t_span=jnp.linspace(0., 10., 200), potentials=[k_pot])

# plotting the trajectory
draw_trajectory_2d(positions)


from surfaces import torus, hyperbolic_paraboloid
from lagrangian import constrained_lagrangian, draw_trajectory

# create a surface
surf = (torus, {'R': 3.0, 'r': 1.0})

# change the fixed_pt dimension for the 3D example
k_pot = (elastic, {'k': 30, 'fixed_pt': jnp.zeros(3)})

# call the lagrangian function
L, eom = constrained_lagrangian(surf, q, q_t, mass, potentials=[k_pot])
print(L)
print(eom)

# evolving the lagrangian
positions, _ = evolve_lagrangian(q, q_t, mass, t_span=jnp.linspace(0., 100., 1000), potentials=[k_pot], surface=surf)

# plotting the trajectory
draw_trajectory(positions, surf)

surf = (hyperbolic_paraboloid, {'a': 0.5, 'b': 0.3})

# evolving the lagrangian
positions, _ = evolve_lagrangian(q, q_t, mass, t_span=jnp.linspace(0., 100., 1000), potentials=[k_pot], surface=surf)

# plotting the trajectory
draw_trajectory(positions, surf)
