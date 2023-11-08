import jax.numpy as jnp



# TorusMap = LocalMap(Torus(3.0, 1.0), Interval(0, 2 * jnp.pi), Interval(0, 2 * jnp.pi))
# TorusMap.draw_surface()

# class KleinBottle(ParametricSurface):
#     def __init__(self, a):
#         super().__init__(
#             lambda u, v: (a + jnp.cos(v / 2) * jnp.sin(u) - jnp.sin(v / 2) * jnp.sin(2 * u)) * jnp.cos(v),
#             lambda u, v: (a + jnp.cos(v / 2) * jnp.sin(u) - jnp.sin(v / 2) * jnp.sin(2 * u)) * jnp.sin(v),
#             lambda u, v: jnp.sin(v / 2) * jnp.sin(u) + jnp.cos(v / 2) * jnp.sin(2 * u),
#             Interval(0, 2 * jnp.pi),
#             Interval(0, 2 * jnp.pi)
#         )


# draw_sphere(radius=1.0, centre=[13.0, 0.0, 0.0])

# def draw_torus(radius, inner_radius, centre):
#     print("radius:", radius)
#     print("inner_radius:", inner_radius)
#     print("centre:", centre)

#     # Create a ParametricTorus object.
#     torus = Torus(radius, inner_radius, centre)

#     # Define the angle parameters of the surface.
#     polar_angle = jnp.linspace(0, 2 * jnp.pi, 100)
#     azimuthal_angle = jnp.linspace(0, 2. * jnp.pi, 100)

#     # Create a meshgrid of the angle parameters.
#     U, V = jnp.meshgrid(polar_angle, azimuthal_angle)

#     # Calculate the surface points.
#     x, y, z = torus(U, V)

#     # Plot the surface using matplotlib.
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

#     # Set axis labels and plot title.
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('Surface of a Torus')

#     plt.show()


# def draw_Klein(R):
# # Define the parameters of the sphere.

#     # Create a Sphere object.
#     bottle = KleinBottle(R)

#     # Define the angle parameters of the surface.
#     polar_angle = jnp.linspace(0, 2 * jnp.pi, 100)
#     azimuthal_angle = jnp.linspace(0, 2 * jnp.pi, 100)

#     # Create a meshgrid of the angle parameters.
#     U, V = jnp.meshgrid(polar_angle, azimuthal_angle)

#     # Calculate the surface points.
#     x, y, z = bottle(U, V)

#     # Plot the surface using matplotlib.
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

#     # Set axis labels and plot title.
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('Surface of a Klein bottle')

#     plt.show()

               
# def animate_klein():
#     from matplotlib.animation import FuncAnimation
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Define the range of 'a' for the animation
#     a_values = jnp.linspace(0.1, 5.0, 100)

#     # Initialize the surface plot
#     U, V = jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, 100), jnp.linspace(0, 2 * jnp.pi, 100))
#     x, y, z = KleinBottle(a_values[0])(U, V)
#     surface = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

#     # Update function for the animation
#     def update(frame):
#         ax.cla()  # Clear the previous frame
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')
#         ax.set_title(f'Surface of a Klein bottle (a={a_values[frame]:.2f})')

#         x, y, z = KleinBottle(a_values[frame])(U, V)
#         surface = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=len(a_values), repeat=False)

#     plt.show()

