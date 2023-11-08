from surface import ParametricSurface, Sphere
from potentials import gravitational_potential, PotentialEnergy
from jax import numpy as jnp
from jax import jacrev
from matplotlib import pyplot as plt

class Lagrangian(object):
    def __init__(self, potential: PotentialEnergy = None) -> None:
        self.potential = potential
        self.mass = None


    def bind_mass(self, mass):
        self.mass = mass
    

    def __call__(self, *args):
        if self.mass is not None:
            T = 0.5 * self.mass * jnp.dot(args[1], args[1])
            V = 0.0 if self.potential is None else self.potential(*args, self.mass)
            return T - V
        else: 
            raise NameError("Before calling the Lagrangian function you need to bind a mass calling self.bind_mass(mass=)")


    def derivatives(self):
        dLdx = jacrev(self.__call__, argnums=0)
        dLdxdot = jacrev(self.__call__, argnums=1)
        dLdt = jacrev(self.__call__, argnums=2)
        return dLdx, dLdxdot, dLdt
    

    def euler_update(self, q, qdot, t):
        dLdx = jacrev(self.__call__, argnums=0)
        q_new = q + t * qdot
        qdot_new = qdot + t * dLdx(q, qdot)
        return q_new, qdot_new
    

    def runge_kutta_update(self, q, qdot, t):
        dLdx = jacrev(self.__call__, argnums=0)

        # Step 1: Compute k1
        k1_qdot = qdot
        k1_qdotdot = dLdx(q, qdot)
        k1_qdot_new = qdot + 0.5 * t * k1_qdotdot

        # Step 2: Compute k2
        k2_qdot = qdot + 0.5 * t * k1_qdotdot
        k2_qdotdot = dLdx(q + 0.5 * t * k1_qdot, k2_qdot)
        k2_qdot_new = qdot + 0.5 * t * k2_qdotdot

        # Step 3: Compute k3
        k3_qdot = qdot + 0.5 * t * k2_qdotdot
        k3_qdotdot = dLdx(q + 0.5 * t * k2_qdot, k3_qdot)
        k3_qdot_new = qdot + t * k3_qdotdot

        # Step 4: Compute k4
        k4_qdot = qdot + t * k3_qdotdot
        k4_qdotdot = dLdx(q + t * k3_qdot, k4_qdot)

        # Update q and qdot using weighted averages of k1, k2, k3, and k4
        q_new = q + (t / 6.0) * (k1_qdot + 2 * k2_qdot + 2 * k3_qdot + k4_qdot)
        qdot_new = qdot + (t / 6.0) * (k1_qdotdot + 2 * k2_qdotdot + 2 * k3_qdotdot + k4_qdotdot)

        return q_new, qdot_new

        


# class ConstrainedLagrangian(Lagrangian):
#     def __init__(self, surface: ParametricSurface, potential: PotentialEnergy = None) -> None:
#         self.map = surface
#         super(ConstrainedLagrangian, self).__init__(potential=potential)

#     def __call__(self, q, qdot, t=None):
#         return self.kinetic_energy(qdot, t) - super().potential_energy(self.map(q), self.map.velocity(q), t)

#     def kinetic_energy(self, t):
#         qdot = self.map.velocity(q)
#         kinetic_matrix = 0.5 * self.mass * self.map.metric(q)
#         return jnp.dot(qdot.T, jnp.dot(kinetic_matrix, qdot))
      

    
# Initial conditions
q0 = jnp.array([0.0, 0.0, 400.0])
qdot0 = jnp.array([30.0, 0.0, 0.0])

# Create empty arrays to store the trajectory
positions = [q0]

# Time step and number of steps
tmax = 10
num_steps = 500
dt = tmax / num_steps

m = 1.0
G_pot = PotentialEnergy(gravitational_potential)

L = Lagrangian(potential=G_pot)
L.bind_mass(m)
print("Lagragian", L(q0, qdot0))

dLdx, dLdxdot, _ = L.derivatives()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([0, 300])  # Set the X-axis limits
ax.set_ylim([-1, 1])  # Set the Y-axis limits
ax.set_zlim([0, 400])  # Set the Z-axis limits

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Title for the plot
ax.set_title('3D Animation', y=1.05)  # Adjust the 'y' parameter to raise the title

# Set the initial time display
current_time = 0.0
time_text = ax.text2D(0.85, 0.05, "", transform=ax.transAxes, fontsize=12, color='black')

# Define the update function for the animation
def update(step):
    global q0, qdot0, current_time
    q_temp, qdot_temp = q0, qdot0

    q0, qdot0 = L.runge_kutta_update(q_temp, qdot_temp, dt)
    positions.append(q0)

    current_time += dt

    # Plot your data here without clearing the axes
    ax.plot3D(*zip(*positions), color='blue')

    # Update and display the current time in the animation
    time_text.set_text(f"Time: {current_time:.3f} seconds")

    return time_text,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=lambda: [time_text])

# Show the animation (you can save it as a video as well)
# plt.show()
ani.save("../media/fall.gif")