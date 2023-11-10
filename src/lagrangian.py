from surface import ParametricSurface
from jax import numpy as jnp
from jax import grad, hessian, jacfwd
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Lagrangian(object):

    def __init__(self, potentials: list[tuple[callable, dict]] = []) -> None:
        self.potentials = potentials


    def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
        V = 0.0
        for pot_fn, pot_params in self.potentials:
            V += pot_fn(q, q_t, mass, **pot_params)
        T = 0.5 * mass * jnp.dot(q_t, q_t)
        return T - V


    def eom(self, q, q_t, mass):
        dLdq = grad(self.__call__, argnums=0)
        dLdq_t = grad(self.__call__, argnums=1)
        H = hessian(self.__call__, argnums=1)(q, q_t, mass)
        q_tt = (jnp.dot(jnp.linalg.pinv(H), dLdq(q, q_t, mass) - jnp.dot(jacfwd(dLdq_t, 0)(q, q_t, mass), q_t)))
        return q_t, q_tt
    
    def integrate_eom(self, q0, q_t0, mass, time_span):
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
    def __init__(self, surface: ParametricSurface, potentials: list[tuple[callable, dict]] = []) -> None:
        self.map = surface
        super(ConstrainedLagrangian, self).__init__(potentials)

    def __call__(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> float:
        V = 0.0
        for pot_fn, pot_params in self.potentials:
            V += pot_fn(self.map(q), self.map.velocity(q, q_t), mass, **pot_params)
        T = 0.5 * mass * jnp.dot(q_t, jnp.dot(self.map.metric(q), q_t))
        return T - V
    
    def draw_trajectory(self, q0, q_t0, mass, t_span) -> None:
        positions, _ = self.integrate_eom(q0, q_t0, mass, t_span)
        self.map.draw_point(positions.T)

    def animate_trajectory(self, q0, q_t0, mass, t_span, save_path=None) -> None:
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