from jax import numpy as jnp
from jax import jacrev


class PotentialEnergy(object):
    def __init__(self, fn, **params):
        self.fn = fn
        self.params = params

    def __call__(self, *args):
        return self.fn(*args, **self.params)
    

def gravitational_potential(x: jnp.array, xdot: jnp.array, mass, G=9.81):
    return mass * G * x[2]