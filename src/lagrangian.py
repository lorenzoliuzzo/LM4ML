from parametric import Surface
from jax import numpy as jnp
from jax import grad, hessian, jacfwd
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Lagrangian(object):
    """
    Lagrangian class for defining and solving the equations of motion in a Lagrangian mechanics system.

    Attributes:
        potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.

    Methods:
        __init__(self, potentials: list[tuple[callable, dict]] = []) -> None:
            Constructor for the Lagrangian class.

        __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
            Calculates the Lagrangian of the system given the generalized coordinates, their velocities, and the mass.

        eom(self, q, q_t, mass):
            Solves the equations of motion for the system.

        integrate_eom(self, q0, q_t0, mass, time_span):
            Integrates the equations of motion over a specified time span.

    """

    def __init__(self, potentials: list[tuple[callable, dict]] = []) -> None:
        """
        Constructor for the Lagrangian class.

        Parameters:
            potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.
        """

        self.potentials = potentials

    def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
        """
        Calculates the Lagrangian of the system given the generalized coordinates, their velocities and the mass.

        Parameters:
            q (jnp.ndarray): Generalized coordinates.
            q_t (jnp.ndarray): Generalized velocities.
            mass: Mass of the system.

        Returns:
            float: The Lagrangian of the system.
        """

        V = 0.0
        for pot_fn, pot_params in self.potentials:
            V += pot_fn(q, q_t, mass, **pot_params)
        T = 0.5 * mass * jnp.dot(q_t, q_t)
        return T - V

    def eom(self, q, q_t, mass):
        """
        Solves the equations of motion for the system.

        Parameters:
            q (jnp.ndarray): Generalized coordinates.
            q_t (jnp.ndarray): Generalized velocities.
            mass: Mass of the system.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing updated velocities and accelerations.
        """

        dLdq = grad(self.__call__, argnums=0)
        dLdq_t = grad(self.__call__, argnums=1)
        H = hessian(self.__call__, argnums=1)(q, q_t, mass)
        q_tt = (jnp.dot(jnp.linalg.pinv(H), dLdq(q, q_t, mass) - jnp.dot(jacfwd(dLdq_t, 0)(q, q_t, mass), q_t)))
        return q_t, q_tt

    def integrate_eom(self, q0, q_t0, mass, time_span):
        """
        Integrates the equations of motion over a specified time span.

        Parameters:
            q0 (jnp.ndarray): Initial generalized coordinates.
            q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
            mass: Mass of the system.
            time_span: Time span for integration.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing integrated generalized coordinates and velocities.
        """

        # Flatten the initial conditions for odeint
        initial_conditions_flat = jnp.concatenate([q0, q_t0])

        # Define a function for odeint
        def dynamics(y, t, *args):
            q, q_t = jnp.split(y, 2)
            q_t, q_tt = self.eom(q, q_t, *args)
            return jnp.concatenate([q_t, q_tt])

        # Use odeint to integrate the equations of motion
        result = odeint(dynamics, initial_conditions_flat, time_span, mass)

        # Reshape the result to get q and q_t separately
        q, q_t = jnp.split(result, 2, axis=-1)

        return q, q_t

class ConstrainedLagrangian(Lagrangian):
    """
    ConstrainedLagrangian class for solving the equations of motion in a Lagrangian mechanics system
    with constraints represented by a Surface.

    Attributes:
        surface (Surface): Parametric surface representing the constraints.

    Methods:
        __init__(self, surface: Surface, potentials: list[tuple[callable, dict]] = []) -> None:
            Constructor for the ConstrainedLagrangian class.

        __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
            Calculates the Lagrangian of the system with constraints.

        draw_trajectory(self, q0, q_t0, mass, t_span) -> None:
            Draws the trajectory of the system with constraints.

        animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
            Animates the trajectory of the system with constraints.

    """

    def __init__(self, surface: Surface, potentials: list[tuple[callable, dict]] = []) -> None:
        """
        Constructor for the ConstrainedLagrangian class.

        Parameters:
            surface (Surface): Parametric surface representing the constraints.
            potentials (list[tuple[callable, dict]]): A list of potential energy functions and their parameters.
        """

        self.map = surface
        super(ConstrainedLagrangian, self).__init__(potentials)

    def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
        """
        Calculates the Lagrangian of the system with constraints.

        Parameters:
            q (jnp.ndarray): Generalized coordinates.
            q_t (jnp.ndarray): Velocities corresponding to the generalized coordinates.
            mass: Mass of the system.

        Returns:
            float: The Lagrangian of the system with constraints.
        """

        V = 0.0
        for pot_fn, pot_params in self.potentials:
            V += pot_fn(self.map(q), self.map.velocity(q, q_t), mass, **pot_params)
        T = 0.5 * mass * jnp.dot(q_t, jnp.dot(self.map.metric(q), q_t))
        return T - V

    def draw_trajectory(self, q0, q_t0, mass, t_span) -> None:
        """
        Draws the trajectory of the system with constraints.

        Parameters:
            q0 (jnp.ndarray): Initial generalized coordinates.
            q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
            mass: Mass of the system.
            t_span: Time span for integration.

        Returns:
            None
        """

        positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)
        self.map.draw_point(positions.T)

    def animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
        """
        Animates the trajectory of the system with constraints.

        Parameters:
            q0 (jnp.ndarray): Initial generalized coordinates.
            q_t0 (jnp.ndarray): Initial velocities corresponding to the generalized coordinates.
            mass: Mass of the system.
            t_span: Time span for integration.
            save_path (str): Optional path to save the animation.

        Returns:
            None
        """

        positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Animation of the motion")

        # Draw the surface
        self.map.draw(ax)

        line, = ax.plot([], [], [], 'o-', lw=2)
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def init():
            # Initialize the plot
            line.set_data([], [])
            line.set_3d_properties([])
            time_text.set_text('')
            ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
            return line, time_text

        def update(frame):
            # Update the plot for each frame
            x, y, z = self.map(positions[frame, :])
            line.set_data(x, y)
            line.set_3d_properties(z)
            time_text.set_text('Time: {:.2f}'.format(t_span[frame]))
            return line, time_text

        anim = FuncAnimation(fig, update, frames=len(t_span), init_func=init, blit=True)

        if save_path is not None:
            anim.save(save_path, fps=30)

        plt.show()
