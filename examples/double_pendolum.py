from src.surfaces import parametrization, double_pendolum
from src.potentials import potential_energy, gravity
from src.lagrangian import lagrangian, lagrangian_eom, evolve_lagrangian
from src.plotting import draw_2D_trajectory, draw_3D_trajectory, animate_2D_trajectory, animate_3D_trajectory 
import numpy as np
from time import time

nbodies = 2
ndim = 2
mass = np.random.random(nbodies)

# setting the generalized coordinates
q = np.random.random((nbodies, ndim))
q_t = np.random.random((nbodies, ndim))

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)

# setting the constraint parametrization
constraint = parametrization(double_pendolum, l1=1.0, l2=3.0)

# setting the potential energy
g_pot = potential_energy(gravity, g=9.81)   

L = lagrangian(q, q_t, mass, potentials=[g_pot], constraint=constraint)
print("L", L)

eom = lagrangian_eom(q, q_t, mass, potentials=[g_pot], constraint=constraint)
print("eom", eom)

# evolving the lagrangian
t0 = 0.0
tmax = 5. * np.pi
npoints = 500
tspan = np.linspace(t0, tmax, npoints)

start = time()
positions, _ = evolve_lagrangian(tspan, q, q_t, mass, potentials=[g_pot], constraint=constraint)
print(f"Evolution finished in {time() - start}")

# draw_2D_trajectory(positions[:, :] % (2.0 * np.pi))
# draw_3D_trajectory(positions, constraint)

# while np.any(positions[:] < 0) or np.any(positions[:] >= 2. * np.pi):
#     positions = positions % (2. * np.pi)
# animate_2D_trajectory(tspan, positions, constraint)

animate_3D_trajectory(tspan, positions, constraint)