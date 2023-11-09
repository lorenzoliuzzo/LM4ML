from surface import Sphere, Torus
from potentials import gravitational, elastic
from lagrangian import ConstrainedLagrangian
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import numpy as jnp


# Initial conditions
q0 = jnp.array([0.5, 0.3]) * jnp.pi
qdot0 = jnp.array([0.1, 0.3])

L = ConstrainedLagrangian(surface=Sphere(radius=1.), potentials=[gravitational, elastic], k=20, fixed_pos=jnp.zeros(3))

# mass
m = 1.0
L.bind_mass(m)

print("Lagragian", L(q0, qdot0))

L.animate_evolution(q0, qdot0, tmax=30, tstep=0.2)