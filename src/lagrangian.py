import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint


def lagrangian(q: list[jnp.array], 
               q_t: list[jnp.array], 
               mass: list[float], 
               potentials: list[callable] = None,
               constraint: callable = None) -> tuple[float, tuple[jnp.array, jnp.array]]:
    """
    Computes the Lagrangian of a system and the equations of motion.

    Parameters:
        q (list[jnp.array]): Generalized coordinates.
        q_t (list[jnp.array]): Generalized velocities.
        mass (list[float]): Mass of the system.
        potentials (list[callable]): List of potential functions and their parameters.
        constraint (callable, optional): Constraint function to parametrize the Lagrangian.

    Returns:
        Tuple[float, Tuple[jnp.array, jnp.array]]:
            - float: Lagrangian of the system.
            - Tuple[jnp.array, jnp.array]: Tuple containing the generalized velocities and accelerations.

    Notes:
        The Lagrangian of the system is computed based on the generalized coordinates, velocities, mass, and
        potential functions. If a constraint function is provided, the Lagrangian is parametrized by the
        constrained coordinates and velocities.

        The equations of motion (EOM) are also computed, providing the generalized velocities and accelerations.
    """
    q = jnp.array(q)
    q_t = jnp.array(q_t)
    mass = jnp.array(mass)
    assert q.shape == q_t.shape
    assert q.shape[0] == len(mass)

    @jax.jit
    def kinetic_energy(q_t) -> float:
        """Compute the kinetic energy of the system."""
        return 0.5 * jnp.sum(mass * jnp.sum(q_t**2, axis=1))
    
    @jax.jit
    def potential_energy(q, q_t) -> float:
        """Compute the potential energy of the system."""
        if potentials is None:
            return 0.0
        else:
            return jnp.sum(jnp.array([pot_fn(q, q_t, mass) for pot_fn in potentials]))
    
    @jax.jit
    def lagrangian_fn(q, q_t) -> float:
        """
        Compute the Lagrangian of the system.

        If a constraint function is provided, the Lagrangian is parametrized by the constrained coordinates and velocities.
        """
        if constraint is None:
            return kinetic_energy(q_t) - potential_energy(q, q_t)
        else:
            x, x_t = jax.jvp(constraint, (q,), (q_t,))
            return kinetic_energy(x_t) - potential_energy(x, x_t)
    
    @jax.jit
    def eom(q, q_t) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the equations of motion (EOM) for the system.

        The EOM provide the generalized velocities and accelerations.
        """
        dLdq = jax.grad(lagrangian_fn, 0)(q, q_t)
        dLdq_t = jax.grad(lagrangian_fn, 1)
        dLdq_t_dq = jax.jacfwd(dLdq_t, 0)(q, q_t)
        H = jax.jacfwd(dLdq_t, 1)(q, q_t)

        # Reshape dLdq_t_dq and q_t to align indices for matrix multiplication
        dLdq_t_dq_flat = jnp.reshape(dLdq_t_dq, (q.size, q.size))
        q_t_flat = jnp.reshape(q_t, (q.size, 1))

        # Reshape H to make it a square matrix
        H_flat = jnp.reshape(H, (q.size, q.size))
        H_flat_inv = jnp.linalg.pinv(H_flat)

        # Ensure that the shapes are compatible for subtraction
        dLdq_flat = jnp.reshape(dLdq, (q.size, 1))

        # Solve for q_tt using the pseudo-inverse
        q_tt_flat = jnp.dot(H_flat_inv, dLdq_flat - jnp.dot(dLdq_t_dq_flat, q_t_flat))

        # Reshape q_tt to the original shape
        q_tt = jnp.reshape(q_tt_flat, q_t.shape)

        return q_t, q_tt

    return lagrangian_fn(q, q_t), eom(q, q_t)


def evolve_lagrangian(t_span: jnp.array,
                      q0: jnp.ndarray, 
                      q_t0: jnp.ndarray, 
                      mass: jnp.array, 
                      potentials: list[callable] = None, 
                      constraint: callable = None) -> tuple[jnp.array, jnp.array]:
    """
    Integrates the equations of motion over a specified time span.

    Parameters:
        t_span: Time span for integration.
        q0 (jnp.ndarray): Initial generalized coordinates.
        q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
        mass: Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters.
        constraint (callable, optional): Constraint function to parametrize the Lagrangian.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing integrated generalized coordinates and velocities.
    """

    # Combine initial conditions
    initial_conditions = jnp.concatenate([q0, q_t0])

    # Define a function for odeint
    @jax.jit
    def dynamics(y, t):
        q, q_t = jnp.split(y, 2)
        L, (q_t, q_tt) = lagrangian(q, q_t, mass, potentials, constraint)
        result = jnp.concatenate([q_t, q_tt])
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions, t_span)

    # Reshape the result to get q and q_t separately
    q, q_t = jnp.split(result, 2, axis=1)

    return q, q_t
