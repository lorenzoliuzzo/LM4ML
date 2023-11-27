import jax
import jax.numpy as jnp


def parametrization(constraint: callable, **kwargs) -> callable:
    """
    Wraps a constraint function with given keyword arguments into a mapping function.

    Parameters:
        constraint (callable): The constraint function to be wrapped.
        **kwargs: Additional keyword arguments to be passed to the constraint function.

    Returns:
        callable: A mapping function that applies the constraint function to its input.

    Notes:
        The returned mapping function can be used in the context of defining constraints for a Lagrangian system.
    """
    @jax.jit
    def map(q: jnp.ndarray):
        return constraint(q, **kwargs)
    return map


@jax.jit
def polar(q: float, norm: float = None) -> jnp.array:
    """
    Converts polar coordinates to Cartesian coordinates.

    Parameters:
        q (jnp.ndarray): Input polar coordinates.
        norm (float, optional): Scaling factor for the output coordinates.

    Returns:
        jnp.ndarray: Cartesian coordinates.

    Notes:
        If the input is a 1D array, it is reshaped to a 2D array.
    """

    @jax.jit
    def x(q: float):
        return jnp.cos(q)

    @jax.jit
    def y(q: float):
        return jnp.sin(q)

    @jax.jit
    def point(q: float):
        return jnp.hstack([x(q), y(q)])

    return norm * point(q) if norm is not None else point(q)


@jax.jit
def spherical(q: jnp.array, norm: float = 1.0) -> jnp.array:
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
        q (jnp.ndarray): Input spherical coordinates.
        norm (float, optional): Scaling factor for the output coordinates.

    Returns:
        jnp.ndarray: Cartesian coordinates.

    """
    @jax.jit
    def x(q):
        return jnp.cos(q[0]) * jnp.sin(q[1])

    @jax.jit
    def y(q):
        return jnp.sin(q[0]) * jnp.sin(q[1])

    @jax.jit
    def z(q):
        return jnp.cos(q[1])

    @jax.jit
    def point(q: jnp.array):
        return  jnp.array([x(q), y(q), z(q)])

    return norm * point(q)


@jax.jit
def circle(q: float, radius: float = 1.0, center: jnp.array = None) -> jnp.array:
    """
    Maps polar coordinates to points on a circle.

    Parameters:
        q (jnp.ndarray): Input polar coordinates.
        radius (float, optional): Radius of the circle.
        center (jnp.ndarray, optional): Center of the circle.

    Returns:
        jnp.ndarray: Points on the circle.

    Notes:
        If center is provided, the points are shifted accordingly.
    """
    r = polar(q, radius)
    return r if center is None else center + r


@jax.jit
def sphere(q: jnp.array, radius: float = 1.0, center: jnp.array = None) -> jnp.array:
    """
    Maps spherical coordinates to points on a sphere.

    Parameters:
        q (jnp.array): Input spherical coordinates.
        radius (float, optional): Radius of the sphere.
        center (jnp.array, optional): Center of the sphere.

    Returns:
        jnp.array: Points on the sphere.

    Notes:
        If center is provided, the points are shifted accordingly.
    """
    r = spherical(q, radius)
    return r if center is None else center + r


@jax.jit
def double_pendulum(q: jnp.ndarray, l1: float, l2: float, fixed_pt: jnp.array = None):
    """
    Maps generalized coordinates of a double pendulum to Cartesian coordinates.

    Parameters:
        q (jnp.ndarray): Generalized coordinates for the two bodies.
        l1 (float): Length of the first pendulum.
        l2 (float): Length of the second pendulum.
        fixed_pt (jnp.array, optional): Fixed point for the first pendulum.

    Returns:
        jnp.ndarray: Cartesian coordinates of the double pendulum.
    """
    assert q.shape[0] == 2 and q.shape[1] <= 2

    if q.shape[1] == 1:
        x1 = circle(q[0], l1, fixed_pt)
        x2 = circle(q[1], l2, x1)
    elif q.shape[1] == 2:
        x1 = sphere(q[0], l1, fixed_pt)
        x2 = sphere(q[1], l2, x1)

    return jnp.vstack([x1, x2])


@jax.jit
def triple_pendulum(q: jnp.ndarray, l1: float, l2: float, l3: float, fixed_pt: jnp.ndarray = None):
    if (q.shape[1] == 1):
        x1 = circle(q[0], l1, fixed_pt)
        x2 = circle(q[1], l2, x1)
        x3 = circle(q[2], l3, x2)
    elif (q.shape[1] == 2):
        x1 = sphere(q[0], l1, fixed_pt)
        x2 = sphere(q[1], l2, x1)
        x3 = sphere(q[2], l3, x2)
    else: 
        raise ValueError("Unsupported dimensions for q. Expected 1 or 2 columns.")

    return jnp.vstack([x1, x2, x3])



# def cone(q: jnp.ndarray, height: float, radius: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return q[:, 0] * radius * (height - q[:, 1]) / height

#     @jax.jit
#     def y(q):
#         return q[:, 0] * radius * (height - q[:, 1]) / height

#     @jax.jit
#     def z(q):
#         return q[:, 0]

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)


# def cylinder(q: jnp.ndarray, height: float, radius: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return radius * jnp.cos(q[:, 0])

#     @jax.jit
#     def y(q):
#         return radius * jnp.sin(q[:, 0])

#     @jax.jit
#     def z(q):
#         return height * q[:, 1]

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)


# def torus(q: jnp.ndarray, R: float, r: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return (R + r * jnp.cos(q[:, 1])) * jnp.cos(q[:, 0])

#     @jax.jit
#     def y(q):
#         return (R + r * jnp.cos(q[:, 1])) * jnp.sin(q[:, 0])

#     @jax.jit
#     def z(q):
#         return r * jnp.sin(q[:, 1])

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)


# def hyperboloid(q: jnp.ndarray, a: float, b: float, c: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return a * q[:, 0]

#     @jax.jit
#     def y(q):
#         return b * q[:, 1]

#     @jax.jit
#     def z(q):
#         return c * jnp.sqrt(1 + (q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2)

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)


# def hyperbolic_paraboloid(q: jnp.ndarray, a: float, b: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return a * q[:, 0]

#     @jax.jit
#     def y(q):
#         return b * q[:, 1]

#     @jax.jit
#     def z(q):
#         return (q[:, 0] / a) ** 2 - (q[:, 1] / b) ** 2

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)


# def ellipsoid(q: jnp.ndarray, a: float, b: float, c: float, center: jnp.array = jnp.zeros(3)):
#     @jax.jit
#     def x(q):
#         return a * q[:, 0]

#     @jax.jit
#     def y(q):
#         return b * q[:, 1]

#     @jax.jit
#     def z(q):
#         return c * jnp.sqrt(1 - (q[:, 0] / a) ** 2 - (q[:, 1] / b) ** 2)

#     @jax.jit
#     def coord(q):
#         return jnp.array([x(q), y(q), z(q)]).T

#     def grad(q):
#         return jax.jacfwd(coord)

#     return center.T + coord(q), grad(q)
