from surface import Sphere
from map import Map
from lagrangian import ConstrainedLagrangian, PotentialEnergy
from jax import numpy as jnp


# mapping a surface in R^3
map = Map(surface=Sphere(radius=1.0))

# set initial condition
map.set_local_coordinates(jnp.array([0.0, 0.5 * jnp.pi]))
map.set_local_velocities(jnp.array([0.0, 1.0]))

# map.draw_surface()

print("position on the map:", map.get_local_coordinates())
print("position on the surface:", map.get_global_coordinates())
print("velocity on the map:", map.get_local_velocities())
print("velocity on the surface:", map.get_global_velocities())

# Create a Lagrangian function on the local map
L = ConstrainedLagrangian(map=map)

# Bind a mass to the Lagrangian 
L.bind_mass(mass=1.0)

# Compute the Jacobians of the Lagrangian with respect to q and qdot
dLdq, dLdqdot, dLdt = L.derivatives()

# aliases for mass point coordinates
q = map.get_local_coordinates()
qdot = map.get_local_velocities()

print("Lagrangian value:", L(q, qdot))
print("dLdq:", dLdq(q, qdot))
print("dLdqdot:", dLdqdot(q, qdot))

print("Lagrangian value:", L(map.get_global_coordinates(), map.get_global_velocities()))
print("dLdq:", dLdq(map.get_global_coordinates(), map.get_global_velocities()))
print("dLdqdot:", dLdqdot(map.get_global_coordinates(), map.get_global_velocities()))


L = ConstrainedLagrangian(map=map, potential=gravitational_potential)
L.bind_mass(mass=1.0)
dLdq, dLdqdot, dLdt = L.derivatives()

print("Lagrangian value:", L(q, qdot))
print("dLdq:", dLdq(q, qdot))
print("dLdqdot:", dLdqdot(q, qdot))
