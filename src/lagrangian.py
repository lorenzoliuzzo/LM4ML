import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt

def lagrangian(q: jnp.ndarray, 
               q_t: jnp.ndarray, 
               mass: jnp.array, 
               potentials: list[tuple[callable, dict]] = [],
               constraint: callable = None) -> tuple[float, tuple[jnp.array, jnp.array]]:
    """
    Computes the Lagrangian of a system and the equations of motion.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass (jnp.array): Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters.
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

    Examples:
        q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        q_t = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        mass = jnp.array([1.0, 2.0])
        potentials = [(potential_func, {"param": value})]
        constraint_func = lambda x: x**2  # Example constraint function

        lagrangian_val, eom_result = lagrangian(q, q_t, mass, potentials, constraint_func)
    """
    assert q.shape == q_t.shape
    assert q.shape[0] == len(mass)

    @jax.jit
    def kinetic_energy(q_t) -> float:
        """Compute the kinetic energy of the system."""
        return 0.5 * jnp.sum(mass * jnp.sum(q_t**2, axis=1))
    
    @jax.jit
    def potential_energy(q, q_t) -> float:
        """Compute the potential energy of the system."""
        return jnp.sum(jnp.array([pot_fn(q, q_t, mass, **pot_params) for pot_fn, pot_params in potentials]))
    
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
        q_tt_flat = H_flat_inv @ (dLdq_flat - dLdq_t_dq_flat @ q_t_flat)

        # Reshape q_tt to the original shape
        q_tt = jnp.reshape(q_tt_flat, q_t.shape)
                            
        return q_t, q_tt

    return lagrangian_fn(q, q_t), eom(q, q_t)


def evolve_lagrangian(t_span,
                      q0: jnp.ndarray, 
                      q_t0: jnp.ndarray, 
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
        L, eom_step = lagrangian(q, q_t, mass, potentials, constraint)
        q_t, q_tt = eom_step
        result = jnp.concatenate([q_t, q_tt])
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions, t_span)

    # Reshape the result to get q and q_t separately
    q, q_t = jnp.split(result, 2, axis=1)

    return q, q_t


def draw_trajectory(q: jnp.ndarray, constraint: tuple[callable, dict] = None) -> None:
    """
    Draws the trajectories of the system without constraints.

    Parameters:
        q0 [jnp.ndarray]: List of initial generalized coordinates for each trajectory.
        q_t0 [jnp.ndarray]: List of initial velocities for each trajectory.
        mass (List): List of masses for each trajectory.
        t_span: Time span for integration.

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    if constraint is not None:
        q = jax.vmap(constraint, in_axes=0, out_axes=0)(q)

    # Iterate over trajectories
    for i in range(q.shape[1]):
        trajectory = q[:, i, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Trajectory {i + 1}')

    ax.legend()
    plt.show()

    # # Iterate over n_bodies
    # for trajectory in q:
    #     ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='.')

    # plt.show()


def draw_trajectory_2d(q: jnp.ndarray, surface: tuple[callable, dict] = None) -> None:

    fig, ax = plt.subplots()
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if surface is not None:
        surf_fn, surf_params = surface
        surf_points = [surf_fn(q_i, **surf_params)[0] for q_i in q]
        q = jnp.array(surf_points)

    # Iterate over trajectories
    for trajectory in q:
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o')

    plt.show()


def animate_trajectory(q: jnp.ndarray, surface: tuple[callable, dict] = None) -> None:
    return