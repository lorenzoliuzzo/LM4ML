from lagrangian import Lagrangian
from potentials import PotentialEnergy, gravitational_potential
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import numpy as jnp
# from mpl_toolkits.mplot3d import Axes3D

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
plt.show()
# ani.save("../media/fall.gif")