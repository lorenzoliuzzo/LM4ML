from jax import numpy as jnp

def gravitational(x: jnp.array, xdot: jnp.array, mass, G=9.81):
    return mass * G * x[2]

def elastic(x: jnp.array, xdot: jnp.array, mass, k, fixed_pos, rest_length=0.0):
    displacement = fixed_pos - x
    return 0.5 * mass * k * jnp.square(jnp.linalg.norm(displacement) - rest_length)