import jax 
from jax import numpy as jnp


def parametrization(constraint: callable, **kwargs) -> callable:
    @jax.jit
    def map(q: jnp.ndarray):
        return constraint(q, **kwargs)
    return map


@jax.jit
def polar_coord(q: jnp.ndarray, norm: float =  1.0) -> jnp.ndarray:
    if q.ndim == 1: # If the input is a 1D array, convert it to a 2D array
        q = q.reshape((1, -1))

    @jax.jit
    def x(q):
        return jnp.cos(q[:])
    @jax.jit
    def y(q):
        return jnp.sin(q[:])
    
    @jax.jit
    def point(q: jnp.array):
        return jnp.array([x(q), y(q)]).T
    
    return norm * point(q)


@jax.jit
def circle(q: jnp.ndarray, radius: float = 1.0, center: jnp.ndarray = None):
    r = polar_coord(q, radius)
    return r if center is None else center + r


@jax.jit
def spherical_coord(q: jnp.ndarray, norm: float =  1.0) -> jnp.ndarray:
    if q.ndim == 1:
        # If the input is a 1D array, convert it to a 2D array
        q = q.reshape((1, -1))

    @jax.jit
    def x(q):
        return jnp.cos(q[:, 0]) * jnp.sin(q[:, 1])
    @jax.jit
    def y(q):
        return jnp.sin(q[:, 0]) * jnp.sin(q[:, 1])
    @jax.jit
    def z(q):
        return jnp.cos(q[:, 1])
    
    @jax.jit
    def point(q: jnp.array):
        return jnp.array([x(q), y(q), z(q)]).T
    
    return norm * point(q)


@jax.jit
def sphere(q: jnp.ndarray, radius: float = 1.0, center: jnp.ndarray = None):
    r = spherical_coord(q, radius)
    return r if center is None else center + r

@jax.jit
def double_pendolum(q: jnp.ndarray, l1: float, l2: float, fixed_pt: jnp.ndarray = None):
    if (q.shape[1] == 1):
        x1 = circle(q[0], l1, fixed_pt)
        x2 = circle(q[1], l2, x1)
    elif (q.shape[1] == 2):
        x1 = sphere(q[0], l1, fixed_pt)
        x2 = sphere(q[1], l2, x1)
    else: 
        raise TypeError("Wrong dimensions")
    
    return jnp.vstack([x1, x2])

@jax.jit
def triple_pendolum(q: jnp.ndarray, l1: float, l2: float, l3: float, fixed_pt: jnp.ndarray = None):
    if (q.shape[1] == 1):
        x1 = circle(q[0], l1, fixed_pt)
        x2 = circle(q[1], l2, x1)
        x3 = circle(q[2], l3, x2)
    elif (q.shape[1] == 2):
        x1 = sphere(q[0], l1, fixed_pt)
        x2 = sphere(q[1], l2, x1)
        x3 = sphere(q[2], l3, x2)
    else: 
        raise TypeError("Wrong dimensions")
    
    return jnp.vstack([x1, x2, x3])

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
