import jax
from jax import numpy as jnp
from scipy import constants
import jax
import jax.numpy as jnp


def potential(pot_fn: callable, **kwargs) -> callable:
    """
    Wraps a potential energy function with given keyword arguments into a potential energy mapping function.

    Parameters:
        pot_fn (callable): The potential energy function to be wrapped.
        **kwargs: Additional keyword arguments to be passed to the potential energy function.

    Returns:
        callable: A mapping function that applies the potential energy function to its input.

    Notes:
        The returned mapping function can be used in the context of defining potential energy for a Lagrangian system.
    """
    @jax.jit
    def fn(q: jnp.ndarray, q_t: jnp.ndarray, mass: jnp.array):
        return pot_fn(q, q_t, mass, **kwargs)
    return fn        

@jax.jit
def gravity(x: jnp.array, x_t: jnp.array, mass: jnp.array, g: float = 9.81) -> float:
    """
    Computes the gravitational potential energy for a system of particles.

    Parameters:
        x (jnp.ndarray): Cartesian coordinates of the particles.
        x_t (jnp.ndarray): Velocities of the particles.
        mass (jnp.array): Masses of the particles.
        g (float, optional): Acceleration due to gravity. Default is 9.81 m/s^2.

    Returns:
        jnp.ndarray: Gravitational potential energy.

    Notes:
        Assumes the last column of x contains the vertical coordinate.
    """
    return g * mass * x[-1]


@jax.jit
def elastic(x: jnp.ndarray, x_t: jnp.ndarray, mass: jnp.array, k: float, l0: float = 0.0, fixed_pt: jnp.ndarray = jnp.zeros(3)):
    """
    Computes the elastic potential energy for a system of particles connected by springs.

    Parameters:
        x (jnp.ndarray): Cartesian coordinates of the particles.
        x_t (jnp.ndarray): Velocities of the particles.
        mass (jnp.array): Masses of the particles.
        k (float): Spring constant.
        l0 (float, optional): Rest length of the springs. Default is 0.0.
        fixed_pt (jnp.ndarray, optional): Coordinates of a fixed point. Default is the origin.

    Returns:
        jnp.ndarray: Elastic potential energy.

    Notes:
        Assumes each row of x corresponds to a particle.
    """
    displacement = x - fixed_pt
    return 0.5 * k * jnp.linalg.norm(displacement)


# @jax.jit
# def gravitational(x: jnp.ndarray, x_t: jnp.ndarray, sources):
#     """
#     Gravitational potential function for multiple bodies.

#     Parameters:
#         q (jnp.ndarray): Generalized coordinates (positions) of shape (n_bodies, n_dim).
#         q_t (jnp.ndarray): Generalized velocities of shape (n_bodies, n_dim).
#         sources (list): List of masses for each body.

#     Returns:
#         float: Gravitational potential energy.
#     """    
    
#     nbodies = len(sources)
#     V = 0.0
#     for i in range(nbodies):
#         for j in range(i + 1, nbodies):
#             if i != j:
#                 V += constants.G * sources[i] * sources[j] / jnp.linalg.norm(x[i] - x[j])
#     return V


# def electric(x: jnp.ndarray, x_t: jnp.ndarray, sources):
#     return sources[0] * sources[1] / ((4.0 * jnp.pi * constants.epsilon_0) * jnp.linalg.norm(x))
