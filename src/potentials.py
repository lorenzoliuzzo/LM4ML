from jax import numpy as jnp

# Example potential function
def gravitational(x: jnp.array, x_t: jnp.array, mass: float, G=9.81):
    return mass * G * x[2]

# Example potential function
def elastic(x: jnp.array, x_t: jnp.array, mass: float, k: float, fixed_pos: jnp.array, rest_length: float =0.0):
    displacement = fixed_pos - x
    return 0.5 * k * jnp.sum(jnp.square(displacement)) 
