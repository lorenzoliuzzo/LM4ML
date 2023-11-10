from parametric import Interval, Surface
from jax import numpy as jnp


class Sphere(Surface):
    # A sphere in R^3.

    def __init__(self, radius = 1.0, centre = jnp.array([0, 0, 0])):
        """
        Parameters
        ----------
        radius  The radius of the sphere.
        centre  The centre of the sphere.
        """
        self.centre = centre
        self.radius = radius
        super().__init__(
            lambda u, v: self.centre[0] + self.radius * jnp.cos(u) * jnp.sin(v),
            lambda u, v: self.centre[1] + self.radius * jnp.sin(u) * jnp.sin(v),
            lambda u, v: self.centre[2] + self.radius * jnp.cos(v),
            Interval(0.0, 2.0*jnp.pi), Interval(0.0, jnp.pi)
        )


class Torus(Surface):
    # A torus in R^3.

    def __init__(self, R, r, centre = jnp.array([0, 0, 0])):
        """ 
        Parameters
        ----------
        centre  The centre of the torus.
        R       The major radius of the torus.
        r       The minor radius of the torus.
        """
        self.centre = centre
        self.R = R
        self.r = r
        super().__init__(
            lambda u, v: self.centre[0] + (self.R + self.r * jnp.cos(v)) * jnp.cos(u),
            lambda u, v: self.centre[1] + (self.R + self.r * jnp.cos(v)) * jnp.sin(u),
            lambda u, v: self.centre[2] + self.r * jnp.sin(v), 
            Interval(0.0, 2.0*jnp.pi), Interval(0.0, 2.0*jnp.pi)
        )


class Hyperboloid(Surface):
    # A hyperboloid in R^3.

    def __init__(self, a, b, centre, interval_u):
        """
        Parameters
        ----------
        a       The major radius of the hyperboloid.
        b       The minor radius of the hyperboloid.
        centre  The centre of the hyperboloid.
        """
        
        self.centre = centre
        self.a = a
        self.b = b
        super().__init__(
            lambda u, v: self.centre[0] + self.a * jnp.cosh(u) * jnp.cos(v),
            lambda u, v: self.centre[1] + self.a * jnp.cosh(u) * jnp.sin(v),
            lambda u, v: self.centre[2] + self.b * jnp.sinh(u),
            interval_u,
            Interval(0, 2 * jnp.pi)
        )
        

class Cone(Surface):

    def __init__(self, apex, base_radius, height):
        """
        A cone in R^3.

        Parameters
        ----------
        apex : Quantity (array-like)
            The apex (tip) of the cone (x, y, z).
        base_radius : Quantity
            The radius of the base of the cone.
        height : Quantity
            The height of the cone.
        """
        self.apex = apex
        self.base_radius = base_radius
        self.height = height
        super().__init__(
            lambda u, v: self.apex[0] + u * self.base_radius * jnp.cos(v),
            lambda u, v: self.apex[1] + u * self.base_radius * jnp.sin(v),
            lambda u, v: self.apex[2] + u * self.height,
            Interval(0, 1),
            Interval(0, 2 * jnp.pi)
        )


class Cylinder(Surface):
    # A cylinder in R^3.

    def __init__(self, r, h):
        """
        Parameters
        ----------
        r   The radius of the cylinder.
        h   The height of the cylinder.
        """

        super().__init__(
            lambda u, v: r * jnp.cos(u),
            lambda u, v: r * jnp.sin(u),
            lambda u, v: v
        )


class EllipticCone(Surface):
    # An elliptic cone in R^3.

    def __init__(self, center, a, b, height):
        """
        Parameters
        ----------
        a       Scaling parameter in the x-direction.
        b       Scaling parameter in the y-direction.
        height  Height of the cone.
        center  The center point of the cone.
        """
        
        self.a = a
        self.b = b
        self.height = height
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * u * jnp.cos(v),
            lambda u, v: self.center[1] + self.b * u * jnp.sin(v),
            lambda u, v: self.center[2] + self.height * u
        )



class EllipticParaboloid(Surface):
    # An elliptic paraboloid in R^3.

    def __init__(self, a, b, center):
        """
        Parameters
        ----------
        a       Scaling parameter in the x-direction.
        b       Scaling parameter in the y-direction.
        center  The center point of the paraboloid.
        """
        self.a = a
        self.b = b
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * u * jnp.cos(v),
            lambda u, v: self.center[1] + self.b * u * jnp.sin(v),
            lambda u, v: self.center[2] + u ** 2
        )


class HyperbolicParaboloid(Surface):
    # A hyperbolic paraboloid in R^3.

    def __init__(self, a, b, center, interval_u, interval_v):
        """
        Parameters
        ----------
        a       Scaling parameter in the x-direction.
        b       Scaling parameter in the y-direction.
        center  The center point of the paraboloid.
        """
        self.a = a
        self.b = b
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * u * jnp.cos(v),
            lambda u, v: self.center[1] + self.b * u * jnp.sin(v),
            lambda u, v: self.center[2] + self.a * u ** 2 - self.b * v ** 2,
            interval_u,
            interval_v
        )


class Ellipsoid(Surface):
    """
    An ellipsoid in R^3.
    """
    def __init__(self, a, b, c, center):
        """
        Parameters
        ----------
        a: Scaling parameter in the x-direction.
        b: Scaling parameter in the y-direction.
        c: Scaling parameter in the z-direction.
        center: The center point of the ellipsoid.
        """
        self.a = a
        self.b = b
        self.c = c
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * jnp.cos(u) * jnp.sin(v),
            lambda u, v: self.center[1] + self.b * jnp.sin(u) * jnp.sin(v),
            lambda u, v: self.center[2] + self.c * jnp.cos(v),
            Interval(0.0, 2.0 * jnp.pi),
            Interval(0.0, jnp.pi)
        )


class MobiusStrip(Surface):
    """
    A Mobius strip in R^3.
    """
    def __init__(self, width, radius):
        """
        Parameters
        ----------
        width: The width of the strip.
        radius: The radius of the strip.
        """
        super().__init__(
            lambda u, v: (radius + u * jnp.cos(v / 2)) * jnp.cos(v),
            lambda u, v: (radius + u * jnp.cos(v / 2)) * jnp.sin(v),
            lambda u, v: u * jnp.sin(v / 2),
            Interval(-width / 2, width / 2),
            Interval(0.0, 2.0 * jnp.pi)
        )

class TrefoilKnot(Surface):
    """
    A trefoil knot in R^3.
    """
    def __init__(self, scale=1.0):
        """
        Parameters
        ----------
        scale: Scaling parameter.
        """
        super().__init__(
            lambda u, v: scale * (jnp.sin(3 * u) + 2 * jnp.sin(2 * u)),
            lambda u, v: scale * (jnp.cos(3 * u) - 2 * jnp.cos(2 * u)),
            lambda u, v: scale * (-jnp.sin(u)),
            Interval(0.0, 2.0 * jnp.pi),
            Interval(0.0, 2.0 * jnp.pi)
        )


class SpiralHelix(Surface):
    """
    A spiral helix in R^3.
    """
    def __init__(self, radius=1.0, pitch=1.0, height=1.0):
        """
        Parameters
        ----------
        radius: Radius of the helix.
        pitch: Pitch of the helix.
        height: Height of the helix.
        """
        super().__init__(
            lambda u, v: radius * jnp.cos(u),
            lambda u, v: radius * jnp.sin(u),
            lambda u, v: pitch * u,
            Interval(0.0, 4.0 * jnp.pi),
            Interval(0.0, height)
        )



class KleinBottle(Surface):
    """
    A Klein bottle in R^3.
    """
    def __init__(self, a=1.0, b=1.0):
        """
        Parameters
        ----------
        a: Scaling parameter.
        b: Scaling parameter.
        """
        super().__init__(
            lambda u, v: (a + b * jnp.cos(v / 2) * jnp.sin(u) - b * jnp.sin(v / 2) * jnp.sin(2 * u)),
            lambda u, v: (b * jnp.sin(v / 2) * jnp.sin(u) + a * jnp.cos(v / 2) * jnp.sin(2 * u)),
            lambda u, v: b * jnp.sin(v / 2) * jnp.cos(u),
            Interval(0.0, 2.0 * jnp.pi),
            Interval(0.0, 2.0 * jnp.pi)
        )


class KleinBottle2(Surface):
    """
    Another parametrization of a Klein bottle in R^3.
    """
    def __init__(self, scale=1.0):
        """
        Parameters
        ----------
        scale: Scaling parameter.
        """
        super().__init__(
            lambda u, v: scale * (-2/15 * jnp.cos(u) * (3 * jnp.cos(v) - 30 * jnp.sin(u) + 90 * jnp.cos(u)**4 * jnp.sin(u) - 60 * jnp.cos(u)**6 * jnp.sin(u) + 5 * jnp.cos(u)**8 * jnp.sin(u))),
            lambda u, v: scale * (-2/15 * jnp.sin(u) * (3 * jnp.cos(v) - 3 * jnp.cos(u)**2 * jnp.cos(v) - 48 * jnp.cos(u)**4 * jnp.cos(v) + 48 * jnp.cos(u)**6 * jnp.cos(v) - 60 * jnp.sin(u) + 5 * jnp.cos(u)**8 * jnp.cos(v) + 5 * jnp.cos(u)**8 * jnp.sin(v))),
            lambda u, v: scale * (1/15 * (3 * jnp.cos(v) - 30 * jnp.sin(u) + 90 * jnp.cos(u)**4 * jnp.sin(u) - 60 * jnp.cos(u)**6 * jnp.sin(u) + 5 * jnp.cos(u)**8 * jnp.sin(u))),
            Interval(0.0, 2.0 * jnp.pi),
            Interval(0.0, 2.0 * jnp.pi)
        )