from typing import Any
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
import timeit

class Interval(object):
    # An interval class to represent a subset of R.

    def __init__(self, start, end):
        """
        Parameters
        ----------
        start   The start of the interval.
        end     The end of the interval.
        """
        if (start > end):
            tmp = end
            end = start
            start = tmp
        self.start = start
        self.end = end

    def contains(self, t):
        return jnp.logical_and(jnp.all(self.start <= t), jnp.all(t <= self.end))
        
    def midpoint(self):
        return 0.5 * (self.end - self.start)


class Surface(object):
    """
    A parametric surface in R^3.

    Attributes:
        parametrization (callable): A function representing the parametrization of the surface.

    Methods:
        __init__(self, parametrization: callable) -> None:
            Constructor for the Surface class.

        __call__(self, q: jnp.ndarray) -> jnp.ndarray:
            Evaluate the parametric surface at the given parameters `q`.

        velocity(self, q: jnp.array, qdot: jnp.array) -> jnp.array:
            Calculate the velocity of the surface at the given parameters `q` and velocities `qdot`.

        coordinate_vectors(self, q: jnp.array) -> jnp.array:
            Compute the coordinate vectors of the surface at the given parameters `q`.

        metric(self, q: jnp.array) -> jnp.array:
            Compute the metric tensor of the surface at the given parameters `q`.

    """

    def __init__(self, parametrization: callable) -> None:
        """
        Constructor for the Surface class.

        Parameters:
            parametrization (callable): A function representing the parametrization of the surface.
        """
        self.parametrization = parametrization
    
    def __call__(self, q: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the parametric surface at the given parameters `q`.

        Parameters:
            q (jnp.ndarray): The parameter values for the surface.

        Returns:
            jnp.ndarray: An array representing the coordinates (x, y, z) of the surface at `q`.
        """
        return self.parametrization(q)

    def velocity(self, q: jnp.array, qdot: jnp.array) -> jnp.array:
        """
        Calculate the velocity of the surface at the given parameters `q` and velocities `qdot`.

        Parameters:
            q (jnp.array): The parameter values for the surface.
            qdot (jnp.array): Velocities corresponding to the parameters `q`.

        Returns:
            jnp.array: The velocity of the surface.
        """
        return jnp.dot(self.coordinate_vectors(q).T, qdot)

    def coordinate_vectors(self, q: jnp.array) -> jnp.array:
        """
        Compute the coordinate vectors of the surface at the given parameters `q`.

        Parameters:
            q (jnp.array): The parameter values for the surface.

        Returns:
            jnp.array: An array containing the coordinate vectors.
        """
        coord_vecs = jacrev(self.parametrization)(q).T

        # Normalize the coordinate vectors
        norm_u = jnp.linalg.norm(coord_vecs[0])
        norm_v = jnp.linalg.norm(coord_vecs[1])

        return jnp.array([coord_vecs[0] / norm_u, coord_vecs[1] / norm_v])

    def metric(self, q: jnp.array) -> jnp.array:
        """
        Compute the metric tensor of the surface at the given parameters `q`.

        Parameters:
            q (jnp.array): The parameter values for the surface.

        Returns:
            jnp.array: The metric tensor of the surface.
        """
        jac_u, jac_v = self.coordinate_vectors(q)

        E = jnp.dot(jac_u, jac_u)
        F = jnp.dot(jac_u, jac_v)
        G = jnp.dot(jac_v, jac_v)

        return jnp.array([[E, F], [F, G]])

    def mesh_grid(self, npoints: int = 100):
        u_vec = jnp.linspace(self.udomain.start, self.udomain.end, npoints)
        v_vec = jnp.linspace(self.vdomain.start, self.vdomain.end, npoints)
        return jnp.array(jnp.meshgrid(u_vec, v_vec))

    def draw(self, ax, npoints: int = 100):
        M = self.mesh_grid(npoints)
        X, Y, Z = self.__call__(M)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def draw_point(self, q, npoints: int = 100):
        print("drawing points on the surface...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.f(q)
        ax.scatter(x, y, z, color='red', marker='o')
        self.draw(ax, npoints)
        plt.show()


# @jax.jit 
# def map(parametrization: callable, q: jnp.ndarray, *args: Any, **kwds: Any):
#     primal = parametrization(q, *args)
#     tangent = jax.jacrev(parametrization, 0)
#     return primal, tangent
    

# q = jnp.array([0.0, 0.9 * jnp.pi])

# print(sphere(q, radius=1.0))
# print(map(sphere, q, radius=1.0))

# @jit
# def tangent_velocity(parametrization: callable, q: jnp.array, q_t: jnp.array) -> jnp.ndarray:
#     coord_vecs = jacrev(parametrization)(q).T

#     # Normalize the coordinate vectors
#     norm_u = jnp.linalg.norm(coord_vecs[0])
#     norm_v = jnp.linalg.norm(coord_vecs[1])

#     coord_vecs = jnp.array([coord_vecs[0] / norm_u, coord_vecs[1] / norm_v])

#     return jnp.dot(coord_vecs, q_t)





repeat_times = 5
times = timeit.repeat(lambda: sphere(q, radius=1.0), number=100000, repeat=repeat_times)
avg_time = sum(times) / repeat_times
print(f"Average sphere time: {avg_time:.6f} seconds")

times = timeit.repeat(lambda: map(sphere, q, radius=1.0), number=100000, repeat=repeat_times)
avg_time = sum(times) / repeat_times
print(f"Average map time: {avg_time:.6f} seconds")