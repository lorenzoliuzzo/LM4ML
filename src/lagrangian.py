import jax
from jax import numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt

@jax.jit
def kinetic_energy(q_t: jnp.ndarray, mass: jnp.ndarray):
    return 0.5 * mass * jnp.linalg.norm(q_t)


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
        T = jnp.sum(jax.jit(kinetic_energy)(q_t, mass))
        V = jnp.sum(jnp.array([pot_fn(q, q_t, mass, **pot_params) for pot_fn, pot_params in self.potentials]))
        return T - V    

    def eom(self, q: jnp.ndarray, q_t: jnp.ndarray, mass) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the equations of motion for the system.

        Parameters:
            q (jnp.ndarray): Generalized coordinates.
            q_t (jnp.ndarray): Generalized velocities.
            mass: Mass of the system.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing updated velocities and accelerations.
        """
        dLdq = jax.jacfwd(self.__call__, 0)(q, q_t, mass)
        dLdq_t_dq = jax.jacfwd(jax.jacfwd(self.__call__, 1), 0)(q, q_t, mass)
        H = jax.hessian(self.__call__, 1)(q, q_t, mass)

        dot1 = jnp.tensordot(dLdq_t_dq, q_t, axes=((2, 3), (0, 1)))
        q_tt = jnp.tensordot(jnp.linalg.pinv(H), dLdq - dot1, axes=((2, 3), (1, 0)))
        
        # print("dLdq", dLdq.shape, dLdq) # correct 
        # print("dLdq_t_dq", dLdq_t_dq.shape, dLdq_t_dq) # correct
        # print("H", H.shape, H) # correct
        # print("dot1", dot1.shape, dot1)
        # print("q_tt", q_tt.shape, q_tt)

        return q_t, q_tt

    def eom_int(self, q0: jnp.ndarray, q_t0: jnp.ndarray, mass, t_span) -> tuple[jnp.ndarray, jnp.ndarray]:
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
            q_t, q_tt = self.eom(q, q_t, mass)
            result = jnp.concatenate([q_t, q_tt])  # Flatten the result
            return result

        # Use odeint to integrate the equations of motion
        result = odeint(dynamics, initial_conditions_flat, t_span)

        # Reshape the result to get q and q_t separately
        q, q_t = jnp.split(result, 2, axis=1)

        return q, q_t


    def draw_trajectory(self, q0: jnp.ndarray, q_t0: jnp.ndarray, mass, t_span) -> None:
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

        positions, _ = self.eom_int(q0, q_t0, mass, t_span)

        # Iterate over n_bodies
        for body in range(len(positions[0])):  
            ax.scatter(positions[:, body, 0], positions[:, body, 1], positions[:, body, 2])

        plt.show()