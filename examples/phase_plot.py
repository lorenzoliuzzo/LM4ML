import os, sys
import time

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from src.constraints import parametrization, double_pendulum, circle
from src.potentials import potential_energy, gravity
from src.lagrangian import evolve_lagrangian
from src.plotting import animate_3D_trajectory, animate_2D_trajectory, double_pendulum_phase_plot

import jax
from jax import numpy as jnp
import numpy as np

# setting the randomness
seed = np.random.randint(0, 1000)
key = jax.random.PRNGKey(seed)

nbodies = 2
npoints = 100
theta = jax.random.uniform(key, (npoints, nbodies, 1), float, 0.0, 2.0 * jnp.pi)
print(theta.shape)

cir = parametrization(circle)
print(cir(theta).shape)

# creating the constraint with parametrization
# constraint = parametrization(double_pendulum, l1=1.0, l2=2.0)
# print(constraint(theta).shape)
