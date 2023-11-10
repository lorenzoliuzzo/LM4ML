from surface import Sphere, Torus
from potentials import gravitational, elastic
from lagrangian import ConstrainedLagrangian
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import numpy as jnp


# Initial conditions
mass = 1.0
q0 = jnp.array([0.0, 0.5]) * jnp.pi
q_t0 = jnp.array([0.1, 0.0])

# setting the potential parameters
gravitational_pot_params = {'G': 9.81}
elastic_pot_params = {'k': 50, 'fixed_pos': jnp.zeros(3)}

# Creating the lagrangian
L = ConstrainedLagrangian(surface=Torus(R=2.0, r=1.), potentials=[(gravitational, gravitational_pot_params), (elastic, elastic_pot_params)])

# plotting the trajectory
# L.draw_trajectory(q0, q_t0, mass, t_span=jnp.linspace(0.0, 20.0, 100))

# animating the trajectory
L.animate_trajectory(q0, q_t0, mass, t_span=jnp.linspace(0.0, 20.0, 500))