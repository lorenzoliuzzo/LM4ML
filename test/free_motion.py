from jax import numpy as jnp
import timeit

from trash.lagrangian import Lagrangian, lagrangian_eom, integrate_lagrangian_eom

# setting the mass of the particle
mass = 1.0

# the vectors passed to the Lagrangian must follow this shape (n_bodies, n_dimension)
x = jnp.array([[0.0, 0.0, 10.0]])
x_t = jnp.array([[1.0, -3.0, -1.0]])

# Creating the Lagrangian without potential energy
L = Lagrangian()

print(lagrangian_eom(L, x, x_t, mass))

# def lagrangian_eval(q, q_t, mass):
#     return L(q=q, q_t=q_t, mass=mass)

# # Evaluating the Lagrangian
# L_value = L(q=x, q_t=x_t, mass=mass)
# print("Lagrangian value:", L_value)

# jit_time = timeit.timeit(lambda: lagrangian_eom(lagrangian_eval, x, x_t, mass), number=10000)
# print(f"JIT eom time: {jit_time:.6f} seconds")

# jit_time = timeit.timeit(lambda: integrate_lagrangian_eom(lagrangian_eval, x, x_t, mass), number=10000)
# print(f"JIT int_eom time: {jit_time:.6f} seconds")

# # Integrate the equation of motion in a selected time span and draw the trajectory
# L.draw_trajectory(q0=x, q_t0=x_t, mass=mass, t_span=jnp.linspace(0.0, 10.0, 200))


# from lagrangian import ConstrainedLagrangian
# from surfaces import Sphere

# # Creating a ConstrainedLagrangian choosing a surface
# L = ConstrainedLagrangian(surface=Sphere(radius=1.0))

# # setting the generical coordinates
# q = jnp.array([[0.0, 0.5 * jnp.pi]]) 
# q_t = jnp.array([[0.5, 0.3]]) 

# # Evaluating the Lagrangian
# L_value = L(q=q, q_t=q_t, mass=mass)
# print("ConstrainedLagrangian value:", L_value)

# timeit.timeit(lambda: lagrangian_eom(L, q, q_t, mass)
# pos, _ = integrate_lagrangian_eom(L, q, q_t, mass, t_span=jnp.linspace(0.0, 1.0, 10))
# Integrate the equation of motion in a selected time span and draw the trajectory on the surface
# L.draw_trajectory(q0=q, q_t0=q_t, mass=mass, t_span=jnp.linspace(0.0, 10.0, 20))