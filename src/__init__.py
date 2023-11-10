import sys
sys.path.append('src')  # Adjust the relative path as needed

from .parametric import Interval, Surface
# from .curve import ParametricCurve, Circle
from .surfaces import Sphere, Cone, Cylinder, Torus, Hyperboloid, HyperbolicParaboloid, Ellipsoid, EllipticCone, EllipticParaboloid, SpiralHelix, MobiusStrip, KleinBottle, KleinBottle2, TrefoilKnot 
from .potentials import gravitational, elastic
from .lagrangian import Lagrangian, ConstrainedLagrangian