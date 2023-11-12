import sys
sys.path.append('src')  # Adjust the relative path as needed

# from ..trash.parametric import Interval, Surface, spherical_coord
# # from .curve import ParametricCurve, Circle
# from ..trash.surfaces import Sphere
# # , Cone, Cylinder, Torus, Hyperboloid, HyperbolicParaboloid, Ellipsoid, EllipticCone, EllipticParaboloid, SpiralHelix, MobiusStrip, KleinBottle, KleinBottle2, TrefoilKnot 
# from .potentials import gravitational, electric, elastic, harmonic

from .lagrangian import Lagrangian
from .potentials import gravity