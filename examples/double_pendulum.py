import os, sys
import time

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from src.constraints import parametrization, double_pendulum
from src.potentials import potential, potential_energy, gravity
from src.lagrangian import lagrangian, lagrangian_eom, evolve_lagrangian
from src.plotting import animate_3D_trajectory, animate_2D_trajectory, double_pendulum_phase_plot

import jax
from jax import numpy as jnp
import numpy as np

# setting the randomness
seed = np.random.randint(0, 1000)
key = jax.random.PRNGKey(seed)

# setting the initial conditions
nbodies = 2
ndim = 2
mass = jnp.ones(nbodies)
q = jax.random.uniform(key, (nbodies, ndim))
q_t = jax.random.uniform(key, (nbodies, ndim))

# creating the constraint with parametrization
constraint = parametrization(double_pendulum, l1=1.0, l2=1.0)

# creating the gravity with potential_energy
g_pot = potential(gravity, g=9.81)   
print("L", lagrangian(jnp.ones((nbodies, ndim)), jnp.ones((nbodies, ndim)), mass, jax.vmap(potential_energy([g_pot]))))
print("L", lagrangian(jnp.ones(nbodies), jnp.ones(nbodies), 1., potential_energy([g_pot])))
# print("L", lagrangian(jnp.ones(2), jnp.ones(2), 1.0, [g_pot]))
# print("L", lagrangian(q, q_t, mass, [g_pot], constraint))

# print("L_eom", lagrangian_eom(q, q_t, mass, [g_pot], constraint))
# print("L_eom", lagrangian_eom(jnp.ones(2), jnp.ones(2), 1.0, [g_pot]))
# print("L_eom", lagrangian_eom(jnp.ones((nbodies, ndim)), jnp.ones((nbodies, ndim)), mass, [g_pot], constraint))

# setting the time evolution parameters
# tmax = 20.
# npoints = 100
# tspan = jnp.linspace(0., tmax, npoints)

# # evolving the lagrangian
# start = time.time()
# positions, velocities = evolve_lagrangian(tspan, q, q_t, mass, potentials=[g_pot], constraint=constraint)
# end = time.time()
# print(f"Evolution finished in {end - start} s")

# animate the trajectory
# draw_2D_trajectory(tspan, positions) #, save_path='../media/double_pendulum.mp4')

# # setting the data directory for saving the trajectory
# data_dir = 'data/dp/' + str(seed)
# os.makedirs(data_dir, exist_ok=True)

# jnp.save(os.path.join(data_dir, 't.npy'), tspan)
# jnp.save(os.path.join(data_dir, 'q.npy'), positions)
# jnp.save(os.path.join(data_dir, 'q_t.npy'), velocities)
