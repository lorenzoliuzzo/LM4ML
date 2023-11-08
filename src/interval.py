from jax import numpy as jnp

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
