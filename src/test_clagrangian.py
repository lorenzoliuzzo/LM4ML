from surfaces import Sphere, Torus, Cone, Ellipsoid, EllipticParaboloid, MobiusStrip
from potentials import gravitational, elastic
from lagrangian import ConstrainedLagrangian
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import numpy as jnp


# Initial conditions
mass = 1.0
q0 = jnp.array([0.0, 0.5]) * jnp.pi
q_t0 = jnp.array([0.1, 0.0])

# Creating the surface
surf = Ellipsoid(a=2.0, b=0.5, c=0.2, center=jnp.zeros(3))

# setting the potential parameters
gravitational_pot_params = {'G': 9.81}
# elastic_pot_params = {'k': 50, 'fixed_pos': jnp.array([0.0, 0.0, 2.0])}

# Creating a potential energy as a list
pot = [(gravitational, gravitational_pot_params)]

# Creating the lagrangian
L = ConstrainedLagrangian(surface=surf, potentials=pot)

# plotting the trajectory
L.draw_trajectory(q0, q_t0, mass, t_span=jnp.linspace(0.0, 5.0, 100))

# animating the trajectory
# L.animate_trajectory(q0, q_t0, mass, t_span=jnp.linspace(0.0, 20.0, 500), save_path="../media/sphere_gravity_spring_NP.gif")