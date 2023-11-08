import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from test.map import *


class LocalMap(object):
    
    def __init__(self, surface: ParametricSurface, udomain: Interval, vdomain: Interval):
        """
        Initialize a local map of a surface.

        Parameters
        ----------
        surface : ParametricSurface
            The parametric surface to map locally.
        u : float
            The local parameter value for u.
        v : float
            The local parameter value for v.
        """
        self.udomain = udomain
        self.vdomain = vdomain
        self.surface = surface
        self.q: jnp.array((2,)) = None
        self.qdot: jnp.array((2,)) = None

    def set_local_coordinates(self, u, v):
        """
        Get the local coordinates (u, v) of the map.

        Returns
        -------
        tuple
            A tuple containing the local u and v coordinates.
        """
        if self.udomain.contains(u):
            if self.vdomain.contains(v):
                self.q = [u, v]
            else:
                raise ValueError(f"v value {v} is outside the domain [{self.vdomain.start}, {self.vdomain.end}]")
        else:
            raise ValueError(f"u value {u} is outside the domain [{self.udomain.start}, {self.udomain.end}]")
    
    def set_local_velocities(self, du, dv):
        self.qdot = jnp.array([du, dv])

    def get_global_coordinates(self):
        return self.surface(self.q)

    def get_global_velocities(self):
        return self.coordinate_vectors() @ self.qdot

    def coordinate_vectors(self):
        return jnp.array(jax.jacrev(self.surface, (0, 1))(self.q[0], self.q[0]))
    

    def metric(self):
        jac_u, jac_v = self.coordinate_vectors()
        E = jnp.dot(jac_u, jac_u)
        F = jnp.dot(jac_u, jac_v)
        G = jnp.dot(jac_v, jac_v)
        return jnp.array([[E, F], [F, G]])


    def mesh_grid(self, npoints: int = 100):
        u_vec = jnp.linspace(self.udomain.start, self.udomain.end, npoints)
        v_vec = jnp.linspace(self.vdomain.start, self.vdomain.end, npoints)
        return jnp.array(jnp.meshgrid(u_vec, v_vec))


    def draw_surface(self):
        M = self.mesh_grid()
        P = self.surface(M[0], M[1])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(P[0], P[1], P[2], cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    
def Gravitational(**kwargs):
    # Calculate potential energy due to gravity
    g = 9.81  # gravitational acceleration (m/s^2)
    mass = kwargs['mass']
    height = kwargs['surface_position'][2]  # z-coordinate
    return mass * g * height
    
# class Elastic(PotentialEnergy):
#     def __init__(self, spring_constant, rest_length):
#         elastic_function = lambda q, rest_length=rest_length, spring_constant=spring_constant: 0.5 * spring_constant * (q - rest_length) ** 2
#         super().__init__(elastic_function)


# class Gravitational(PotentialEnergy): 
#     def __init__(self, constant):
#         gravitational_function = lambda q, mass, constant=constant: mass * constant / jnp.linalg.norm(q)
#         super().__init__(gravitational_function)


class Lagrangian(object):

    def __init__(self, map: LocalMap, potentials = None) -> None:
        self.map = map
        self.potentials = [potentials] if potentials is not None else potentials
    
    def __call__(self, **kwargs):
        return self.kinetic_energy(**kwargs) - self.potential_energy(**kwargs)

    def kinetic_energy(self, mass, **kwargs):
        T = 0.0
        if 'qdot' in kwargs:
            T += jnp.linalg.norm(self.map.metric() * kwargs['qdot'])**2
        T *= 0.5 * mass
        return T
     
    def potential_energy(self, **kwargs):
        U = 0.0
        if self.potentials is not None:
            for pot in self.potentials:
                U += pot(**kwargs)
        return U
        

# sphere_map = LocalMap(surface=Sphere(radius=1), udomain=Interval(0, 2 * jnp.pi), vdomain=Interval(0, jnp.pi))
# sphere_map.set_local_coordinates(jnp.pi / 2, jnp.pi / 2)
# print(sphere_map.q)

# print(sphere_map.get_global_coordinates())

# metric = sphere_map.metric()
# print('metrica\n', metric)
# print('det', jnp.linalg.det(metric))
# print('inv metrica\n', jnp.linalg.inv(metric))

# sphere_map.draw_surface()

# q = jnp.array([jnp.pi/2, jnp.pi/2])
# qdot = jnp.array([3.0, 0.1])
# time = 0.0

# lagrangian = Lagrangian(map=sphere_map)
# L = lagrangian(mass=1.0, q=q, qdot=qdot, time=time)
# print(L)

# dLdq = jax.jacrev(lagrangian, (0, 1), q)
# dLdqdot = jax.jacrev(lagrangian, (0, 1), qdot)

# import numpy as np

# sphere_map = LocalMap(surface=Sphere(radius=1), udomain=Interval(0, 2 * jnp.pi), vdomain=Interval(0, jnp.pi))
# pendulum = Lagrangian(sphere_map, potentials=[Gravitational])

# # Set the initial conditions for the pendulum
# u0 = jnp.pi / 4.0  # Initial angle
# v0 = jnp.pi / 2.0  # Initial angular velocity
# du0 = np.random.randn()
# dv0 = np.random.randn()

# sphere_map.set_local_coordinates(u0, v0)
# sphere_map.set_local_velocities(du0, dv0)

# # Define the mass of the pendulum bob
# mass = 1.0  # kg

# # Compute the Lagrangian for the initial conditions
# q0 = sphere_map.q
# qdot0 = sphere_map.qdot
# L0 = pendulum(mass=mass, q=q0, qdot=qdot0)

# from scipy.optimize import minimize

# # Perform optimization (simulate motion) to find new coordinates
# result = minimize(L0, q0, args=(qdot0,))
# q_final = result.x
# qdot_final = result.y

# # Set the new coordinates and calculate the Lagrangian for the final state
# sphere_map.q = q_final
# sphere_map.qdot = qdot_final  # Assume angular velocity is conserved
# final_lagrangian = pendulum(q_final, qdot_final)

# print("Initial Lagrangian:", L0)
# print("Final Lagrangian:", final_lagrangian)
