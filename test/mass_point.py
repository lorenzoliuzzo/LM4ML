from jax import numpy as jnp

class MassPoint(object):

    def __init__(self, map, mass, position, velocity, potential):
        self.map = map
        self.mass = mass
        self.map.set_local_coordinates(position)
        self.map.set_local_velocities(velocity)
        self.potential = potential

    def set_position(self, q):
        self.map.set_local_coordinates(q)

    def set_velocity(self, qdot):
        self.map.set_local_velocities(qdot)

    def get_position(self):
        self.map.get_local_coordinates()

    def get_velocity(self):
        self.map.get_local_velocities()

    def kinetic_matrix(self):
        return self.mass * self.map.metric()
    
    def kinetic_energy(self):
        A = self.kinetic_matrix()
        return 0.5 * jnp.dot(self.get_velocity(), jnp.dot(A, self.get_velocity()))
    
    def potential_energy(self):
        return self.potential(self)

    def lagrangian(self):
        return self.kinetic_energy() - self.potential_energy()
    