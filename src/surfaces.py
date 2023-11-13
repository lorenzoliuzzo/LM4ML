import jax 
from jax import numpy as jnp


def spherical_coord(q: jnp.ndarray, norm: float):
    @jax.jit
    def x(q):
        return norm * jnp.cos(q[:, 0]) * jnp.sin(q[:, 1])
    @jax.jit
    def y(q):
        return norm * jnp.sin(q[:, 0]) * jnp.sin(q[:, 1])
    @jax.jit
    def z(q):
        return norm * jnp.cos(q[:, 1])
    
    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T
    def grad(q):
        return jax.jacfwd(coord)
    
    return coord(q), grad(q)


def sphere(q: jnp.ndarray, radius: float, center: jnp.array = jnp.zeros(3)):
    coord, grad = spherical_coord(q, radius)
    return center.T + coord, grad

def cone(q: jnp.ndarray, height: float, radius: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return q[:, 0] * radius * (height - q[:, 1]) / height

    @jax.jit
    def y(q):
        return q[:, 0] * radius * (height - q[:, 1]) / height

    @jax.jit
    def z(q):
        return q[:, 0]

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)


def cylinder(q: jnp.ndarray, height: float, radius: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return radius * jnp.cos(q[:, 0])

    @jax.jit
    def y(q):
        return radius * jnp.sin(q[:, 0])

    @jax.jit
    def z(q):
        return height * q[:, 1]

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)


def torus(q: jnp.ndarray, R: float, r: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return (R + r * jnp.cos(q[:, 1])) * jnp.cos(q[:, 0])

    @jax.jit
    def y(q):
        return (R + r * jnp.cos(q[:, 1])) * jnp.sin(q[:, 0])

    @jax.jit
    def z(q):
        return r * jnp.sin(q[:, 1])

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)


def hyperboloid(q: jnp.ndarray, a: float, b: float, c: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return a * q[:, 0]

    @jax.jit
    def y(q):
        return b * q[:, 1]

    @jax.jit
    def z(q):
        return c * jnp.sqrt(1 + (q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2)

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)


def hyperbolic_paraboloid(q: jnp.ndarray, a: float, b: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return a * q[:, 0]

    @jax.jit
    def y(q):
        return b * q[:, 1]

    @jax.jit
    def z(q):
        return (q[:, 0] / a) ** 2 - (q[:, 1] / b) ** 2

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)


def ellipsoid(q: jnp.ndarray, a: float, b: float, c: float, center: jnp.array = jnp.zeros(3)):
    @jax.jit
    def x(q):
        return a * q[:, 0]

    @jax.jit
    def y(q):
        return b * q[:, 1]

    @jax.jit
    def z(q):
        return c * jnp.sqrt(1 - (q[:, 0] / a) ** 2 - (q[:, 1] / b) ** 2)

    @jax.jit
    def coord(q):
        return jnp.array([x(q), y(q), z(q)]).T

    def grad(q):
        return jax.jacfwd(coord)

    return center.T + coord(q), grad(q)
