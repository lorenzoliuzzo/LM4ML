import jax
from jax import numpy as jnp
from surfaces import double_pendolum, parametrization
from potentials import gravity
from lagrangian import lagrangian, evolve_lagrangian, draw_trajectory
import numpy.random as rangen

nbodies = ndim = 2
mass = jnp.array(rangen.random(nbodies))

# setting the generalized coordinates
q = jnp.array(rangen.random((nbodies, ndim)))
q_t = jnp.array(rangen.random((nbodies, ndim)))

print("q", q) 
print("q_t", q_t)

# setting the constraint
constraint = parametrization(double_pendolum, l1=10.0, l2=2.0)

# setting the potential
g_pot = (gravity, {})

L = lagrangian(q, q_t, mass, potentials=[g_pot], constraint=constraint)
print("L", L)

# evolving the lagrangian
t0 = 0.0
tmax = 10 * jnp.pi
npoints = 200
tspan = jnp.linspace(t0, tmax, npoints)
positions, _ = evolve_lagrangian(tspan, q, q_t, mass, potentials=[g_pot], constraint=constraint)

# plotting the trajectory
draw_trajectory(positions, constraint)