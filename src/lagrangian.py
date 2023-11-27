import jax
from jax import numpy as jnp


def lagrangian(q: jnp.array, 
               qdot: jnp.array, 
               mass: jnp.array,             
               potentials: list[callable] = None,
               constraint: callable = None) -> jnp.array:
    """
    Computes the Lagrangian of a system and the equations of motion.

    Parameters:
        q (jnp.array): Generalized coordinates.
        qdot (jnp.array): Generalized velocities.
        mass (float): Mass of the system.
        potentials (list[callable]): List of potential functions.
        constraint (callable, optional): Constraint function to parametrize the coordinates.

    Returns:
        float: Lagrangian of the system.

    Notes:
        The Lagrangian of the system is computed based on the generalized coordinates, velocities, mass, and
        potential functions. If a constraint function is provided, the Lagrangian is parametrized by the
        constrained coordinates and velocities.
    """
    assert q.shape == qdot.shape 

    @jax.jit
    def kinetic_energy(qdot, mass) -> jnp.array:
        """Compute the kinetic energy of the system."""
        qdot_norm = jnp.linalg.norm(qdot, axis=-1)
        return 0.5 * mass * jnp.dot(qdot_norm, qdot_norm)

    @jax.jit
    def potential_energy(q, qdot, mass) -> jnp.array:
        """Compute the potential energy of the system."""
        return jnp.sum(jnp.array([pot_fn(q, qdot, mass) for pot_fn in potentials])) if potentials is not None else 0.0

    @jax.jit
    def lagrangian_fn(q, qdot, mass) -> jnp.array:
        if constraint is None:
            return kinetic_energy(qdot, mass) - potential_energy(q, qdot, mass)
        else:
            x, xdot = jax.jvp(constraint, (q,), (qdot,))
            return kinetic_energy(xdot, mass) - potential_energy(x, xdot, mass)
        
    return lagrangian_fn(q, qdot, mass) 