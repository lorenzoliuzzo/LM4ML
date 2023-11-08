from interval import Interval
import jax.numpy as jnp


class ParametricCurve(object):
    # A parametric curve in R^2.

    def __init__(self, x, y, interval: Interval):
        """
        Parameters
        ----------
        x : function
            A function representing the x-coordinate of the curve.
        y : function
            A function representing the y-coordinate of the curve.
        domain : Interval
            The domain of the curve.
        """
        self.x = x
        self.y = y
        self.domain = interval

    def __call__(self, t):
        """
        Evaluate the parametric curve at the given parameter t.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve.

        Returns
        -------
        numpy.array
            An array representing the (x, y) coordinates of the curve at t.

        Raises
        ------
        ValueError
            If t is not within the specified domain.
        """
        if self.domain.contains(t):
            return jnp.array([self.x(t), self.y(t)])
        else: 
            raise ValueError(f"t value {t} is outside the domain [{self.domain.start}, {self.domain.end}]")

    def parametrization(self):
        return self.x, self.y
    

class Circle(ParametricCurve):

    def __init__(self, radius = 1.0, centre = jnp.array([0.0, 0.0])):
        self.centre = centre
        self.radius = radius
        super().__init__(
            lambda t: self.centre[0] + self.radius * jnp.cos(t),
            lambda t: self.centre[1] + self.radius * jnp.sin(t),
            Interval(0.0, 2.0*jnp.pi)
        )
