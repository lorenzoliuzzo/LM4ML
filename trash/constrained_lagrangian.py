


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

#         V = 0.0
#         for pot_fn, pot_params in self.potentials:
#             V += pot_fn(self.map(q), self.map.velocity(q, q_t), mass, **pot_params)
#         T = 0.5 * mass * jnp.dot(q_t, jnp.dot(self.map.metric(q), q_t))
#         return T - V

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

#         positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)
#         self.map.draw_point(positions.T)

#     def animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
#         """
#         Animates the trajectory of the system with constraints.

#         Parameters:
#             q0 (jnp.ndarray): Initial generalized coordinates.
#             q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
#             mass: Mass of the system.
#             t_span: Time span for integration.
#             save_path (str): Optional path to save the animation.

#         Returns:
#             None
#         """

#         positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_title("Animation of the motion")

#         # Draw the surface
#         self.map.draw(ax)

#         line, = ax.plot([], [], [], 'o-', lw=2)
#         time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

#         def init():
#             # Initialize the plot
#             line.set_data([], [])
#             line.set_3d_properties([])
#             time_text.set_text('')
#             ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
#             return line, time_text

#         def update(frame):
#             # Update the plot for each frame
#             x, y, z = self.map(positions[frame, :])
#             line.set_data(x, y)
#             line.set_3d_properties(z)
#             time_text.set_text('Time: {:.2f}'.format(t_span[frame]))
#             return line, time_text

#         anim = FuncAnimation(fig, update, frames=len(t_span), init_func=init, blit=True)

#         if save_path is not None:
#             anim.save(save_path, fps=30)

#         plt.show()