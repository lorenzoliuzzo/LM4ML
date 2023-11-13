import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt


def lagrangian(q: jnp.ndarray, 
               q_t: jnp.ndarray, 
               mass: jnp.array, 
               potentials: list[tuple[callable, dict]] = []) -> tuple[float, tuple[jnp.array, jnp.array]]:
    """
    Computes the Lagrangian of a system.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass (jnp.array): Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters.

    Returns:
        float: Lagrangian of the system.
    """
    assert q.shape == q_t.shape
    assert q.shape[0] == len(mass)

    @jax.jit
    def kinetic_energy(q_t) -> float:
        return 0.5 * jnp.sum(mass * jnp.sum(q_t**2, axis=1))
    
    @jax.jit
    def potential_energy(q, q_t) -> float:
        return jnp.sum(jnp.array([pot_fn(q, q_t, mass, **pot_params) for pot_fn, pot_params in potentials]))
    
    @jax.jit
    def lagrangian_fn(q, q_t) -> float:
        return kinetic_energy(q_t) - potential_energy(q, q_t)
    
    @jax.jit
    def eom(q, q_t) -> tuple[jnp.ndarray, jnp.ndarray]:
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


def constrained_lagrangian(surface: tuple[callable, dict], 
                           q: jnp.ndarray,
                           q_t: jnp.ndarray, 
                           mass: jnp.array, 
                           potentials: list[tuple[callable, dict]] = []) -> tuple[float, tuple[jnp.array, jnp.array]]:
    nbodies = len(mass)
    assert q.shape == q_t.shape
    assert q.shape[0] == nbodies

    surf_fn, surf_params = surface 

    def kinetic_energy(q, q_t, jac) -> float:
        J = jac(q).transpose((0, 2, 1, 3))

        # Compute the metric tensor using einsum       
        G = jnp.einsum('ijkl,ijkl->ij', J, J)

        return 0.5 * jnp.sum(q_t.T @ (jnp.diag(mass) @ G) @ q_t)
    
    @jax.jit
    def potential_energy(x, x_t) -> float:
        return jnp.sum(jnp.array([pot_fn(x, x_t, mass, **pot_params) for pot_fn, pot_params in potentials]))
    
    def surface_velocity(q, q_t, jac) -> jnp.ndarray:
        J = jac(q).transpose(0, 2, 1, 3).reshape(-1, q_t.shape[0] * (q_t.shape[1]))
        surf_pt_t = jnp.dot(J, q_t.flatten())
        return surf_pt_t.reshape(-1, 3)
        
    def lagrangian_fn(q, q_t) -> float:
        surf_pt, surf_jac = surf_fn(q, **surf_params) 
        return kinetic_energy(q, q_t, surf_jac) - potential_energy(surf_pt, surface_velocity(q, q_t, surf_jac))

    def eom(q, q_t) -> tuple[jnp.ndarray, jnp.ndarray]:
        surf_pt, surf_jac = surf_fn(q, **surf_params) # map q to x
        J = surf_jac(q)

        # Compute gradients wrt to q_t
        H_q_t = jax.hessian(kinetic_energy, 1)(q, q_t, surf_jac)
        H_inv = jnp.linalg.pinv(H_q_t)
        J_T_q_t = jax.jacfwd(kinetic_energy, 1)

        # Compute gradients wrt to q
        J_V_q = jax.jacfwd(lagrangian_fn, 0)(q, q_t)
        J_T_q_t_q = jax.jacfwd(J_T_q_t, 0)(q, q_t, surf_jac)

        dot = jnp.tensordot(J_T_q_t_q, q_t, axes=((2, 3), (0, 1)))
        q_tt = jnp.tensordot(H_inv, J_V_q - dot, axes=((2, 3), (1, 0)))        
        return (q_t, q_tt)

    return lagrangian_fn(q, q_t), eom(q, q_t)


def evolve_lagrangian(q0: jnp.ndarray, 
                      q_t0: jnp.ndarray, 
                      mass: jnp.array, 
                      t_span, 
                      potentials: list[tuple[callable, dict]] = [], 
                      surface: tuple[callable, dict] = None):
    """
    Integrates the equations of motion over a specified time span.

    Parameters:
        q0 (jnp.ndarray): Initial generalized coordinates.
        q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
        mass: Mass of the system.
        t_span: Time span for integration.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing integrated generalized coordinates and velocities.
    """

    # Combine initial conditions
    initial_conditions = jnp.concatenate([q0, q_t0])

    # Define a function for odeint
    def dynamics(y, t):
        q, q_t = jnp.split(y, 2)

        if surface is None:
            L, eom_step = lagrangian(q, q_t, mass, potentials)
        else:
            L, eom_step = constrained_lagrangian(surface, q, q_t, mass, potentials)

        q_t, q_tt = eom_step
        result = jnp.concatenate([q_t, q_tt])
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions, t_span)

    # Reshape the result to get q and q_t separately
    q, q_t = jnp.split(result, 2, axis=1)

    return q, q_t


def draw_trajectory(q: jnp.ndarray, surface: tuple[callable, dict] = None) -> None:
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

    if surface is not None:
        surf_fn, surf_params = surface
        surf_points = []
        for i in range(q.shape[0]):
            x, _ = surf_fn(q[i], **surf_params)
            surf_points.append(x)
        q = jnp.array(surf_points)

    # Iterate over n_bodies
    for trajectory in q:
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='.')

    plt.show()


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