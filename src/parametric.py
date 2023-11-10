from jax import numpy as jnp
from jax import jacrev
from matplotlib import pyplot as plt

class Interval(object):
    # An interval class to represent a subset of R.

    def __init__(self, start, end):
        """
        Parameters
        ----------
        start   The start of the interval.
        end     The end of the interval.
        """
        if (start > end):
            tmp = end
            end = start
            start = tmp
        self.start = start
        self.end = end

    def contains(self, t):
        return jnp.logical_and(jnp.all(self.start <= t), jnp.all(t <= self.end))
        
    def midpoint(self):
        return 0.5 * (self.end - self.start)


class Surface(object):
    # A parametric surface in R^3.

    def __init__(self, x, y, z, udomain: Interval, vdomain: Interval):
        """
        Parameters
        ----------
        x : function
            A function representing the x-coordinate of the surface.
        y : function
            A function representing the y-coordinate of the surface.
        z : function
            A function representing the z-coordinate of the surface.
        """
        self.x = x
        self.y = y
        self.z = z
        self.udomain = udomain
        self.vdomain = vdomain
    

    def __call__(self, q: jnp.array):
        """
        Evaluate the parametric surface at the given parameters u and v.

        Parameters
        ----------
        u : float
            The parameter value for u.
        v : float
            The parameter value for v.

        Returns
        -------
        numpy.array
            An array representing the (x, y, z) coordinates of the surface at (u, v).

        Raises
        ------
        ValueError
            If u or v is not within the specified domains.
        """
        u, v = q

        return jnp.array([self.x(u, v), self.y(u, v), self.z(u, v)])
        # if self.udomain.contains(u):
        #     if self.vdomain.contains(v):
        #         return jnp.array([self.x(u, v), self.y(u, v), self.z(u, v)])
        #     else:
        #         raise ValueError(f"v value {v} is outside the domain [{self.vdomain.start}, {self.vdomain.end}]")
        # else: 
        #     raise ValueError(f"u value {u} is outside the domain [{self.udomain.start}, {self.udomain.end}]")


    def velocity(self, q: jnp.array, qdot: jnp.array):   
        return jnp.dot(self.coordinate_vectors(q).T, qdot)


    def coordinate_vectors(self, q: jnp.array):
        coord_vecs = jacrev(self.__call__)(q).T

        # Normalize the coordinate vectors
        norm_u = jnp.linalg.norm(coord_vecs[0])
        norm_v = jnp.linalg.norm(coord_vecs[1])

        return jnp.array([coord_vecs[0] / norm_u, coord_vecs[1] / norm_v])


    def metric(self, q: jnp.array):
        jac_u, jac_v = self.coordinate_vectors(q)

        E = jnp.dot(jac_u, jac_u)
        F = jnp.dot(jac_u, jac_v)
        G = jnp.dot(jac_v, jac_v)

        return jnp.array([[E, F], [F, G]])


    def parametrization(self):
        return jnp.array([self.x, self.y, self.z])
        

    def mesh_grid(self, npoints: int = 100):
        u_vec = jnp.linspace(self.udomain.start, self.udomain.end, npoints)
        v_vec = jnp.linspace(self.vdomain.start, self.vdomain.end, npoints)
        return jnp.array(jnp.meshgrid(u_vec, v_vec))


    def draw(self, ax, npoints: int = 100):
        M = self.mesh_grid(npoints)
        X, Y, Z = self.__call__(M)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def draw_point(self, q, npoints: int = 100):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.__call__(q)
        ax.scatter(x, y, z, color='red', marker='o')
        self.draw(ax, npoints)
        plt.show()