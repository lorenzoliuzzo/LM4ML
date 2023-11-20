import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint

def hamiltonian(q: jnp.ndarray,
                p: jnp.ndarray,
                mass: jnp.array,
                potentials: list[tuple[callable, dict]] = [],
                constraint: callable = None) -> tuple[float, tuple[jnp.array, jnp.array]]:
    """
    Computes the Hamiltonian of a system and the equations of motion.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        p (jnp.ndarray): Generalized momenta.
        mass (jnp.array): Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters.
        constraint (callable, optional): Constraint function to parametrize the Hamiltonian.

    Returns:
        Tuple[float, Tuple[jnp.array, jnp.array]]:
            - float: Hamiltonian of the system.
            - Tuple[jnp.array, jnp.array]: Tuple containing the generalized velocities and accelerations.

    Notes:
        The Hamiltonian of the system is computed based on the generalized coordinates, momenta, mass, and
        potential functions. If a constraint function is provided, the Hamiltonian is parametrized by the
        constrained coordinates and momenta.

        The equations of motion (EOM) are also computed, providing the generalized velocities and accelerations.

    Examples:
        q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        p = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        mass = jnp.array([1.0, 2.0])
        potentials = [(potential_func, {"param": value})]
        constraint_func = lambda x: x**2  # Example constraint function

        hamiltonian_val, eom_result = hamiltonian(q, p, mass, potentials, constraint_func)
    """
    assert q.shape == p.shape
    assert q.shape[0] == len(mass)

    @jax.jit
    def kinetic_energy(p) -> float:
        """Compute the kinetic energy of the system."""
        return 0.5 * jnp.sum(jnp.sum(p**2, axis=1) / mass)

    @jax.jit
    def potential_energy(q) -> float:
        """Compute the potential energy of the system."""
        return jnp.sum(jnp.array([pot_fn(q, p, mass, **pot_params) for pot_fn, pot_params in potentials]))

    @jax.jit
    def hamiltonian_fn(q, p) -> float:
        """
        Compute the Hamiltonian of the system.

        If a constraint function is provided, the Hamiltonian is parametrized by the constrained coordinates and momenta.
        """
        if constraint is None:
            return kinetic_energy(p) + potential_energy(q)
        else:
            x, x_p = jax.jvp(constraint, (q,), (p,))
            return kinetic_energy(x_p) + potential_energy(x)

    @jax.jit
    def eom(q, p) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the equations of motion (EOM) for the system.

        The EOM provide the generalized velocities and accelerations.
        """
        dHdq, dHdp = jax.grad(hamiltonian_fn, (0, 1))(q, p)

        return dHdp, -dHdq

    return hamiltonian_fn(q, p), eom(q, p)


def evolve_hamiltonian(t_span,
                       q0: jnp.ndarray, 
                       p0: jnp.ndarray, 
                       mass: jnp.array, 
                       potentials: list[tuple[callable, dict]] = [], 
                       constraint: tuple[callable, dict] = None):
    """
    Integrates the equations of motion over a specified time span.

    Parameters:
        t_span: Time span for integration.
        q0 (jnp.ndarray): Initial generalized coordinates.
        q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
        mass: Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters.
        constraint (callable, optional): Constraint function to parametrize the Hamiltonian.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing integrated generalized coordinates and velocities.
    """

    # Combine initial conditions
    initial_conditions = jnp.concatenate([q0, p0])

    # Define a function for odeint
    @jax.jit
    def dynamics(y, t):
        q, p = jnp.split(y, 2)
        H, eom_step = hamiltonian(q, p, mass, potentials, constraint)
        q_t, p_t = eom_step
        result = jnp.concatenate([q_t, p_t])
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions, t_span)

    # Reshape the result to get q and p separately
    q, p = jnp.split(result, 2, axis=1)

    return q, p

