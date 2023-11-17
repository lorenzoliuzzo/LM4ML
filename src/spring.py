import jax
from jax import numpy as jnp
from lagrangian import lagrangian, evolve_lagrangian, draw_trajectory, draw_trajectory_2d
from potentials import elastic, gravity
from surfaces import parametrization, sphere

# setting the ic
m = 1.0
x0 = [0.5 * jnp.pi, 0.2 * jnp.pi]
x_t0 = [0.3, -0.1]

# Convert to JAX arrays
q = jnp.array([x0])
q_t = jnp.array([x_t0])
mass = jnp.array([m])

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)


# create a surface
surf = parametrization(sphere, radius=9.81)
point = surf(q)
# # change the fixed_pt dimension for the 3D example
# k_pot = (elastic, {'k': 40, 'fixed_pt': jnp.zeros(3)})

# call the lagrangian function
g_pot = (gravity,{})
L = lagrangian(q, q_t, mass, potentials=[g_pot], constraint=surf)
print(L)

# evolving the lagrangian
t0 = 0.0
tmax = 10 * jnp.pi
npoints = 400
tspan = jnp.linspace(t0, tmax, npoints)
positions, _ = evolve_lagrangian(tspan, q, q_t, mass, potentials=[g_pot], surface=surf)

p0 = surf(positions[0])
p1 = surf(positions[npoints])
print("initial point", p0)
print("final point", p1)
print("total displacement", p1 - p0)

# plotting the trajectory
draw_trajectory(positions, surf)