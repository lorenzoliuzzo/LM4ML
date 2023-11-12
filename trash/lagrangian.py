from trash.parametric import Surface
from jax import numpy as jnp
from jax import grad, hessian, jacrev, jacfwd, vmap, jit
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import timeit

@jit 
def lagrangian(q: jnp.ndarray, q_t: jnp.ndarray, mass, potentials: list[tuple[callable, dict]] = []):
    """
    Calculates the Lagrangian of the system given the generalized coordinates, generalized velocities, mass and potential energies.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass: Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters

    Returns:
        float: The Lagrangian of the system.
    """
    T = 0.5 * jnp.sum(jnp.dot(jnp.dot(q_t, mass * jnp.eye(len(q_t[0]))).T, q_t))        
    V = jnp.sum(jnp.array([pot_fn(q, q_t, mass, **pot_params) for pot_fn, pot_params in potentials])) 
    return T - V


@jit
def lagrangian_eom(q: jnp.ndarray, q_t: jnp.ndarray, mass, potentials: list[tuple[callable, dict]] = []) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves the equations of motion for the system.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass: Mass of the system.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing updated velocities and accelerations.
    """

    dLdq = jacfwd(lagrangian, 0)(q, q_t, mass, potentials)
    dLdq_t_dq = jacfwd(jacfwd(lagrangian, 1), 0)(q, q_t, mass, potentials)
    H = hessian(lagrangian, 1)(q, q_t, mass, potentials)

    dot1 = jnp.tensordot(dLdq_t_dq, q_t, axes=((2, 3), (0, 1)))
    q_tt = jnp.tensordot(jnp.linalg.pinv(H), dLdq - dot1, axes=((2, 3), (1, 0)))

    return q_t, q_tt


@jit
def integrate_lagrangian_eom(q0: jnp.ndarray, q_t0: jnp.ndarray, mass, t_span, potentials: list[tuple[callable, dict]] = []):
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
    # Flatten the initial conditions for odeint
    initial_conditions_flat = jnp.concatenate([q0, q_t0])

    # Define a function for odeint
    def dynamics(y, t):
        q, q_t = jnp.split(y, 2)
        q_t, q_tt = lagrangian_eom(q, q_t, mass, potentials)
        result = jnp.concatenate([q_t, q_tt])  # Flatten the result
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions_flat, t_span)

    # Reshape the result to get q and q_t separately
    q, q_t = jnp.split(result, 2, axis=1)

    return q, q_t


@jit 
def constrained_lagrangian(surf: Surface, q: jnp.ndarray, q_t: jnp.ndarray, mass, potentials: list[tuple[callable, dict]] = []):
    """
    Calculates the Lagrangian of the system given the generalized coordinates, generalized velocities, mass and potential energies.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass: Mass of the system.
        potentials (list[tuple[callable, dict]]): List of potential functions and their parameters

    Returns:
        float: The Lagrangian of the system.
    """
    return lagrangian(surf(q.T), surf.velocity(q.T, q_t.T), mass, potentials)


@jit
def constrained_lagrangian_eom(surf: Surface, q: jnp.ndarray, q_t: jnp.ndarray, mass, potentials: list[tuple[callable, dict]] = []) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves the equations of motion for the system.

    Parameters:
        q (jnp.ndarray): Generalized coordinates.
        q_t (jnp.ndarray): Generalized velocities.
        mass: Mass of the system.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing updated velocities and accelerations.
    """

    dLdq = jacfwd(constrained_lagrangian, 0)(surf, q, q_t, mass, potentials)
    dLdq_t_dq = jacfwd(jacfwd(constrained_lagrangian, 1), 0)(surf, q, q_t, mass, potentials)
    H = hessian(constrained_lagrangian, 1)(surf, q, q_t, mass, potentials)

    dot1 = jnp.tensordot(dLdq_t_dq, q_t, axes=((2, 3), (0, 1)))
    q_tt = jnp.tensordot(jnp.linalg.pinv(H), dLdq - dot1, axes=((2, 3), (1, 0)))

    return q_t, q_tt


@jit
def integrate_constrained_lagrangian_eom(surf: Surface, q0: jnp.ndarray, q_t0: jnp.ndarray, mass, t_span, potentials: list[tuple[callable, dict]] = []):
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
    # Flatten the initial conditions for odeint
    initial_conditions_flat = jnp.concatenate([q0, q_t0])

    # Define a function for odeint
    def dynamics(y, t):
        q, q_t = jnp.split(y, 2)
        q_t, q_tt = constrained_lagrangian(surf, q, q_t, mass, potentials)
        result = jnp.concatenate([q_t, q_tt])  # Flatten the result
        return result

    # Use odeint to integrate the equations of motion
    result = odeint(dynamics, initial_conditions_flat, t_span)

    # Reshape the result to get q and q_t separately
    q, q_t = jnp.split(result, 2, axis=1)

    return q, q_t


# setting the mass of the particle
mass = 1.0

# the vectors passed to the Lagrangian must follow this shape (n_bodies, n_dimension)
x = jnp.array([[0.0, 0.0, 10.0]])
x_t = jnp.array([[1.0, -3.0, -1.0]])
print(lagrangian(x, x_t, mass))
print(lagrangian_eom(x, x_t, mass))
t = timeit.timeit(lambda: integrate_lagrangian_eom(x, x_t, mass, t_span=jnp.linspace(0, 10., 1000)), number=100000)
print(t)


# class Lagrangian(object):
#     """
#     Lagrangian class for defining and solving the equations of motion in a Lagrangian mechanics system.

#     Attributes:
#         potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.

#     Methods:
#         __init__(self, potentials: list[tuple[callable, dict]] = []) -> None:
#             Constructor for the Lagrangian class.

#         __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
#             Calculates the Lagrangian of the system given the generalized coordinates, their velocities, and the mass.

#         eom(self, q, q_t, mass):
#             Solves the equations of motion for the system.

#         integrate_eom(self, q0, q_t0, mass, time_span):
#             Integrates the equations of motion over a specified time span.

#     """

#     def __init__(self, potentials: list[tuple[callable, dict]] = []) -> None:
#         """
#         Constructor for the Lagrangian class.

#         Parameters:
#             potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.
#         """
#         self.potentials = potentials

#     def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
#         """
#         """

#         T = 0.5 * jnp.sum(jnp.dot(q_t.T, jnp.dot(mass * jnp.eye(q_t.ndim), q_t)))        
#         V = jnp.sum(jnp.array([pot_fn(q, q_t, mass, **pot_params) for pot_fn, pot_params in self.potentials])) 
#         return T - V



#     # def animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
#     #     """
#     #     Animates the trajectories of the system without constraints.

#     #     Parameters:
#     #         q0 (List[jnp.ndarray]): List of initial generalized coordinates for each trajectory.
#     #         q_t0 (List[jnp.ndarray]): List of initial velocities for each trajectory.
#     #         mass (List): List of masses for each trajectory.
#     #         t_span: Time span for integration.
#     #         save_path (str): Optional path to save the animation.

#     #     Returns:
#     #         None
#     #     """

#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(111, projection='3d')
#     #     ax.set_title("Animation of the motion")
#     #     ax.set_xlabel("X-axis")
#     #     ax.set_ylabel("Y-axis")
#     #     ax.set_zlabel("Z-axis")

#     #     lines = [ax.plot([], [], [], 'o-', lw=2)[0] for _ in q0]
#     #     time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

#     #     def init():
#     #         # Initialize the plot
#     #         for line in lines:
#     #             line.set_data([], [])
#     #             line.set_3d_properties([])
#     #         time_text.set_text('')
#     #         return lines + [time_text]

#     #     def update(frame):
#     #         # Update the plot for each frame
#     #         updates = []
#     #         for line, q0, q_t0, mass in zip(lines, q0, q_t0, mass):
#     #             positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)
#     #             x, y, z = positions[frame, 0], positions[frame, 1], positions[frame, 2]
#     #             line.set_data(x, y)
#     #             line.set_3d_properties(z)
#     #             updates.append(line)
#     #         time_text.set_text('Time: {:.2f}'.format(t_span[frame]))
#     #         return updates + [time_text]

#     #     anim = FuncAnimation(fig, update, frames=len(t_span), init_func=init, blit=True)

#     #     if save_path is not None:
#     #         anim.save(save_path, fps=30)

#     #     plt.show()


# class ConstrainedLagrangian(Lagrangian):
#     """
#     ConstrainedLagrangian class for solving the equations of motion in a Lagrangian mechanics system
#     with constraints represented by a Surface.

#     Attributes:
#         surface (Surface): Parametric surface representing the constraints.

#     Methods:
#         __init__(self, surface: Surface, potentials: list[tuple[callable, dict]] = []) -> None:
#             Constructor for the ConstrainedLagrangian class.

#         __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
#             Calculates the Lagrangian of the system with constraints.

#         draw_trajectory(self, q0, q_t0, mass, t_span) -> None:
#             Draws the trajectory of the system with constraints.

#         animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
#             Animates the trajectory of the system with constraints.

#     """

#     def __init__(self, surface: Surface, potentials: list[tuple[callable, dict]] = []) -> None:
#         """
#         Constructor for the ConstrainedLagrangian class.

#         Parameters:
#             surface (Surface): Parametric surface representing the constraints.
#             potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.
#         """

#         self.map = surface
#         super(ConstrainedLagrangian, self).__init__(potentials)

#     def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
#         """
#         Calculates the Lagrangian of the system with constraints.

#         Parameters:
#             q (jnp.ndarray): Generalized coordinates.
#             q_t (jnp.ndarray): Velocities corresponding to the generalized coordinates.
#             mass: Mass of the system.

#         Returns:
#             float: The Lagrangian of the system with constraints.
#         """
#         return super().__call__(self.map(q.T), self.map.velocity(q.T, q_t.T), mass)
    
#     def draw_trajectory(self, q0, q_t0, mass, t_span) -> None:
#         """
#         Draws the trajectory of the system with constraints.

#         Parameters:
#             q0 (jnp.ndarray): Initial generalized coordinates.
#             q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
#             mass: Mass of the system.
#             t_span: Time span for integration.

#         Returns:
#             None
#         """

#         positions, _ = integrate_lagrangian_eom(self.__call__, q0, q_t0, mass, t_span)
#         self.map.draw_point(positions.T)

#     # def animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
#     #     """
#     #     Animates the trajectory of the system with constraints.

#     #     Parameters:
#     #         q0 (jnp.ndarray): Initial generalized coordinates.
#     #         q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
#     #         mass: Mass of the system.
#     #         t_span: Time span for integration.
#     #         save_path (str): Optional path to save the animation.

#     #     Returns:
#     #         None
#     #     """

#     #     positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)

#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(111, projection='3d')
#     #     ax.set_title("Animation of the motion")

#     #     # Draw the surface
#     #     self.map.draw(ax)

#     #     line, = ax.plot([], [], [], 'o-', lw=2)
#     #     time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

#     #     def init():
#     #         # Initialize the plot
#     #         line.set_data([], [])
#     #         line.set_3d_properties([])
#     #         time_text.set_text('')
#     #         ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
#     #         return line, time_text

#     #     def update(frame):
#     #         # Update the plot for each frame
#     #         x, y, z = self.map(positions[frame, :])
#     #         line.set_data(x, y)
#     #         line.set_3d_properties(z)
#     #         time_text.set_text('Time: {:.2f}'.format(t_span[frame]))
#     #         return line, time_text

#     #     anim = FuncAnimation(fig, update, frames=len(t_span), init_func=init, blit=True)

#     #     if save_path is not None:
#     #         anim.save(save_path, fps=30)

#     #     plt.show()
