from surface import ParametricSurface
import inspect
from jax import numpy as jnp
from jax import jacrev, jit
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Lagrangian(object):
    def __init__(self, potentials: list[callable] = [], **params) -> None:
        self.potentials = potentials
        self.params = params
        self.mass = None

    def bind_mass(self, mass):
        self.mass = mass    

    def __call__(self, *args):
        if self.mass is not None:
            T = 0.5 * self.mass * jnp.dot(args[1], args[1])
            V = 0.0
            for potential in self.potentials:
                potential_params = {param: self.params[param] for param in inspect.signature(potential).parameters if param in self.params}
                V += potential(*args, mass=self.mass, **potential_params)
            return T - V
        else: 
            raise NameError("Before calling the Lagrangian function you need to bind a mass calling self.bind_mass(mass=)")

    # @jit
    def euler_step(self, q, qdot, dt):
        dLdx = jacrev(self.__call__, argnums=0)
        q_new = q + dt * qdot
        qdot_new = qdot + dt * dLdx(q, qdot)
        return q_new, qdot_new

    # @jit
    def runge_kutta_step(self, q, qdot, dt):
        dLdx = jacrev(self.__call__, argnums=0)

        # Step 1: Compute k1
        k1_qdotdot = dLdx(q, qdot)
        k1_qdot_new = qdot + 0.5 * dt * k1_qdotdot

        # Step 2: Compute k2
        k2_qdotdot = dLdx(q + 0.5 * dt, qdot + 0.5 * dt * k1_qdotdot)
        k2_qdot_new = qdot + 0.5 * dt * k2_qdotdot

        # Step 3: Compute k3
        k3_qdotdot = dLdx(q + 0.5 * dt, qdot + 0.5 * dt * k2_qdotdot)
        k3_qdot_new = qdot + dt * k3_qdotdot

        # Step 4: Compute k4
        k4_qdotdot = dLdx(q + dt, qdot + dt * k3_qdotdot)

        # Update q and qdot using weighted averages of k1, k2, k3, and k4
        q_new = q + (dt / 6.0) * (qdot + 2 * (k1_qdot_new + k2_qdot_new) + k3_qdot_new + k4_qdotdot)
        qdot_new = qdot + (dt / 6.0) * (k1_qdotdot + 2 * k2_qdotdot + 2 * k3_qdotdot + k4_qdotdot)

        return q_new, qdot_new

    # @jit
    def evolve(self, q, qdot, tmax, tstep):
        start_time = time.time()

        positions = [q]
        velocities = [qdot]

        for _ in range(int(tmax / tstep)):
            q, qdot = self.runge_kutta_step(q, qdot, tstep)
            positions.append(q)
            velocities.append(qdot)

        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time} seconds")

        return jnp.array(positions), jnp.array(velocities)

    def animate_evolution(self, q0, qdot0, tmax, tstep):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Animation of the motion', y=1.05)

        current_time = 0.0
        time_text = ax.text2D(0.85, 0.05, "", transform=ax.transAxes, fontsize=12, color='black')

        positions = [q0]

        def update(frame):
            nonlocal q0, qdot0, current_time
            q_temp, qdot_temp = q0, qdot0

            q0, qdot0 = self.runge_kutta_step(q_temp, qdot_temp, tstep)
            positions.append(q0)

            current_time += tstep

            ax.cla()
            ax.plot3D(*zip(*positions), color='blue')

            time_text.set_text(f"Time: {current_time:.3f} seconds")

        ani = FuncAnimation(fig, update, frames=int(tmax / tstep), init_func=lambda: [time_text])

        plt.show()


class ConstrainedLagrangian(Lagrangian):
    def __init__(self, surface: ParametricSurface, potentials: list[callable] = [], **params) -> None:
        self.map = surface
        super(ConstrainedLagrangian, self).__init__(potentials, **params)

    def __call__(self, *args):
        if self.mass is not None:
            T = 0.5 * self.mass * jnp.dot(args[1], jnp.dot(self.map.metric(args[0]), args[1]))
            V = 0.0
            for potential in self.potentials:
                potential_params = {param: self.params[param] for param in inspect.signature(potential).parameters if param in self.params}
                V += potential(self.map(args[0]), self.map.velocity(args[0], args[1]), self.mass, **potential_params)
            return T - V
        else: 
            raise NameError("Before calling the Lagrangian function you need to bind a mass calling self.bind_mass(mass=)")
    

    def animate_evolution(self, q0, qdot0, tmax, tstep):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Constrained Lagrangian Animation', y=1.05)

        # Plot the surface
        self.map.draw(ax)

        current_time = 0.0
        time_text = ax.text2D(0.85, 0.05, "", transform=ax.transAxes, fontsize=12, color='black')

        # Initialize the point plot
        point, = ax.plot3D([], [], [], color='blue', marker='o')

        def update(frame):
            nonlocal q0, qdot0, current_time
            q_temp, qdot_temp = q0, qdot0

            q0, qdot0 = self.runge_kutta_step(q_temp, qdot_temp, tstep)

            # Update the point on the surface plot
            point.set_data(*self.map(q0)[:2])
            point.set_3d_properties(self.map(q0)[2])

            current_time += tstep

            time_text.set_text(f"Time: {current_time:.3f} seconds")

            return point, time_text

        ani = FuncAnimation(fig, update, frames=int(tmax / tstep), init_func=lambda: [point, time_text])

        plt.show()