import jax
from jax import numpy as jnp
from scipy import constants

@jax.jit
def gravity(x: jnp.ndarray, x_t: jnp.ndarray, mass: jnp.array, g: float = 9.81) -> jnp.ndarray:
    return g * jnp.sum(jnp.dot(jnp.diag(mass), x[:, 2]))

@jax.jit
def elastic(x: jnp.ndarray, x_t: jnp.ndarray, mass: jnp.array, k: float, l0: float = 0.0, fixed_pt: jnp.ndarray = jnp.zeros(3)):
    displacement = x - fixed_pt
    return 0.5 * k * jnp.linalg.norm(displacement)


@jax.jit
def gravitational(x: jnp.ndarray, x_t: jnp.ndarray, sources):
    """
    Gravitational potential function for multiple bodies.

    Parameters:
        q (jnp.ndarray): Generalized coordinates (positions) of shape (n_bodies, n_dim).
        q_t (jnp.ndarray): Generalized velocities of shape (n_bodies, n_dim).
        sources (list): List of masses for each body.

    Returns:
        float: Gravitational potential energy.
    """    
    
    nbodies = len(sources)
    V = 0.0
    for i in range(nbodies):
        for j in range(i + 1, nbodies):
            if i != j:
                V += constants.G * sources[i] * sources[j] / jnp.linalg.norm(x[i] - x[j])
    return V


def electric(x: jnp.ndarray, x_t: jnp.ndarray, sources):
    return sources[0] * sources[1] / ((4.0 * jnp.pi * constants.epsilon_0) * jnp.linalg.norm(x))
