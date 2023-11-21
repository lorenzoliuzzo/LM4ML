import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_2D_trajectory(q: jnp.ndarray, constraint: callable = None) -> None:
    """
    Draws the trajectories of the 2D system.

    Parameters:
        q (np.ndarray): Array of generalized coordinates for each trajectory.
        constraint (callable, optional): Constraint function to apply to the trajectories.

    Returns:
        None
    """
    if constraint is None:
        assert q.shape[2] == 2
    else:
        assert q.shape[2] == 1
        q = jax.vmap(constraint, in_axes=0, out_axes=0)(q)

    fig, ax = plt.subplots()
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Iterate over trajectories
    for i in range(q.shape[1]):
        trajectory = q[:, i, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'Trajectory {i + 1}')

    ax.legend()
    plt.show()


def draw_3D_trajectory(q: jnp.ndarray, constraint: callable = None) -> None:
    """
    Draws the trajectories of the 3D system.

    Parameters:
        q [jnp.ndarray]: Array of generalized coordinates for each trajectory.
        constraint (callable, optional): Constraint function to apply to the trajectories.

    Returns:
        None
    """
    if constraint is None:
        assert q.shape[2] == 3
    else:
        assert q.shape[2] == 2
        q = jax.vmap(constraint, in_axes=0, out_axes=0)(q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Iterate over trajectories
    for i in range(q.shape[1]):
        trajectory = q[:, i, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Trajectory {i + 1}')

    ax.legend()
    plt.show()


def animate_2D_trajectory(tspan, q, constraint=None):
    if constraint is None:
        assert q.shape[2] == 2
    else:
        assert q.shape[2] == 1
        q = jax.vmap(constraint, in_axes=0, out_axes=0)(q)

    def update(frame, q, lines):
        for i, line in enumerate(lines):
            line.set_data(q[:frame, i, 0], q[:frame, i, 1])
        time_text.set_text(f'Time: {tspan[frame]:.2f} s')
        return lines + [time_text]
    
    fig, ax = plt.subplots()
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_xlim(jnp.min(q[:, :, 0]), jnp.max(q[:, :, 0]))
    ax.set_ylim(jnp.min(q[:, :, 1]), jnp.max(q[:, :, 1]))

    lines = [ax.plot([], [], label=f'Trajectory {i + 1}')[0] for i in range(q.shape[1])]

    ax.legend()

    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes)

    ani = FuncAnimation(fig, update, frames=q.shape[0], fargs=(q, lines), blit=True, interval=50)
    plt.show()


def animate_3D_trajectory(tspan, q, constraint=None, save_path=None):
    if constraint is None:
        assert q.shape[2] == 3
    else:
        assert q.shape[2] == 2
        q = jax.vmap(constraint, in_axes=0, out_axes=0)(q)
                
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Trajectories of the motion")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim(jnp.min(q[:, :, 0]), jnp.max(q[:, :, 0]))
    ax.set_ylim(jnp.min(q[:, :, 1]), jnp.max(q[:, :, 1]))
    ax.set_zlim(jnp.min(q[:, :, 2]), jnp.max(q[:, :, 2]))

    lines = [ax.plot([], [], [], label=f'Trajectory {i + 1}')[0] for i in range(q.shape[1])]

    ax.legend()

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def update(frame, q, lines):
        for i, line in enumerate(lines):
            line.set_data(q[:frame, i, 0], q[:frame, i, 1])
            line.set_3d_properties(q[:frame, i, 2])
        time_text.set_text(f'Time: {tspan[frame]:.2f} s')
        return lines + [time_text]
    
    ani = FuncAnimation(fig, update, frames=q.shape[0], fargs=(q, lines), blit=True, interval=50)

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=30)
    else:
        plt.show()
