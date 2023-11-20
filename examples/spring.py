import numpy as np
from src.lagrangian import lagrangian, lagrangian_eom, evolve_lagrangian
from src.potentials import potential_energy, elastic, gravity
from src.surfaces import parametrization, sphere
from src.plotting import draw_3D_trajectory, animate_3D_trajectory
from time import time

# setting the ic
nbodies = 3
ndim = 2
mass = np.random.random(nbodies)
q = np.random.random((nbodies, ndim))
q_t = np.random.random((nbodies, ndim))

print("q", q.shape, q)
print("q_t", q_t.shape, q_t)
print("mass", mass.shape, mass)

# create a surface
surf = parametrization(sphere, radius=3.0)

# create a potential energy
k_pot = potential_energy(elastic, k=40, fixed_pt=np.ones(3))
g_pot = potential_energy(gravity)

# call the lagrangian function
L = lagrangian(q, q_t, mass, potentials=[k_pot, g_pot], constraint=surf)
eom = lagrangian_eom(q, q_t, mass, potentials=[k_pot, g_pot], constraint=surf)
print("L", L)
print("eom", eom)

# evolving the lagrangian
t0 = 0.0
tmax = 10 * np.pi
npoints = 500
tspan = np.linspace(t0, tmax, npoints)

start = time()
positions, _ = evolve_lagrangian(tspan, q, q_t, mass, potentials=[k_pot, g_pot], constraint=surf)
end = time()
print(f"Evolution finished in {end - start}")

draw_3D_trajectory(positions, surf)
animate_3D_trajectory(tspan, positions, surf)