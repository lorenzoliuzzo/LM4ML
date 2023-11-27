import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint
from scipy.integrate import solve_ivp

from lagrangian import lagrangian
from potentials import potential, gravity


def lagrangian_eom(q: jnp.array, 
                   qdot: jnp.array, 
                   mass: jnp.array,
                   potentials: list[callable] = None,
                   constraint: callable = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the equations of motion (EOM) for a given lagrangian system.
    
    Parameters:
        q (list[jnp.array]): Initial generalized coordinates.
        qdot (list[jnp.array]): Initial velocities corresponding to the generalized coordinates.
        potentials (list[callable]): List of potential functions.
        constraint (callable, optional): Constraint function to parametrize the coordinates.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing derived velocities and accelerations corresponding to the generalized coordinates.
    """

    @jax.jit
    def lagrangian_fn(q, qdot) -> float:
        """Compute the Lagrangian of the system."""
        return jnp.sum(lagrangian(q, qdot, mass, potentials, constraint))
    
    @jax.jit
    def derivatives(q, qdot) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the partial derivatives of the Lagrangian."""
        dL_dq = jax.grad(lagrangian_fn, 0)(q, qdot)
        dL_dqdot = jax.grad(lagrangian_fn, 1)
        dL_dqdot_dq = jax.jacfwd(dL_dqdot, 0)(q, qdot)
        H = jax.jacfwd(dL_dqdot, 1)(q, qdot)
        return dL_dq, dL_dqdot_dq, H
    
    @jax.jit
    def qddot(q, qdot) -> jnp.array:
        """Compute the equations of motion by solving the linear system H qddot = dL_dq - dL_dqdot_dq qdot"""
        dL_dq, dL_dqdot_dq, H = derivatives(q, qdot)
        qddot = jnp.dot(jnp.linalg.pinv(H), dL_dq - jnp.dot(dL_dqdot_dq, qdot))
        return qddot
    
    return qddot(q, qdot)


def lagrangian_dynamics(t, state, mass, potentials):
    q, qdot = jnp.split(state, 2)    
    qddot = lagrangian_eom(q, qdot, mass, potentials)
    return jnp.concatenate([qdot, qddot])


def evolve_dynamics(dynamics, t0, t, state, mass, potentials):
    sol = solve_ivp(dynamics, (t0, t), state, args=(mass, potentials))
    return sol.y


def lagrangian_param_dynamics(t, state, mass, potentials, params):
    q, qdot = jnp.split(state, 2)

    @jax.jit
    def param_eom(params, q, qdot):
        return lagrangian_eom(q, qdot, mass, potentials)
    
    qddot = param_eom(params, q, qdot)
    return jnp.concatenate([qdot, qddot])


def augmented_dynamics(t, state, mass, potentials):
    """Compute the equations of motion for the adjoint state."""
    q, qdot, adjoint, params = jnp.split(adjoint_state, 4)
    
    @jax.jit
    def dynamics(t, q, qdot, params):
        return lagrangian_eom(q, qdot, mass, potentials)
    
    qddot = dynamics(t, q, qdot, params)
    # df_dt = jax.jacrev(dynamics, 0)(t, q, qdot, params)
    df_dqdot = jax.jacrev(dynamics, 2)(t, q, qdot, params)
    df_dparams = jax.jacrev(dynamics, 3)(t, q, qdot, params)

    return jnp.concatenate([qdot, qddot, -adjoint.T @ df_dqdot]), -adjoint.T @ df_dparams

# print(lagrangian_dynamics(t=0.0, state=jnp.array([0.0, 3.0, 1.0, 0.0]), mass=jnp.ones(1), potentials=[potential(gravity)]))
# print(solve_ivp(lagrangian_dynamics, (0.0, 1.0), jnp.array([0.0, 0.0, 1.0, 0.0]), args=(jnp.ones(1), [potential(gravity)])).y)

q = jnp.zeros(3)
qdot = jnp.ones(3)
print(evolve_dynamics(lagrangian_dynamics, 0.0, 150.0, jnp.concatenate([q, qdot]), jnp.ones(1), [potential(gravity)]).shape)
# adjoint = jnp.zeros(3)
# params = jnp.ones(100)
# adjoint_state = jnp.array([q, qdot, adjoint, params])
# print(adjoint_state.shape)
# print(augmented_dynamics(t=0.0, state=adjoint_state, mass=jnp.ones(1), potentials=[potential(gravity)]))
# print(solve_ivp(augmented_dynamics, (0.0, 1.0), (adjoint_state, params), args=(jnp.ones(1), [potential(gravity)])))


def evolve_lagrangian_adjoints(t : float, 
                               t0: float, 
                               q0: jnp.array,
                               qdot0: jnp.array,
                               mass: jnp.array,
                               **kwargs):
    """
    Compute the adjoint Lagrangian equations of motion (EOM) for the adjoint state.

    Parameters:
        params (jnp.array): System parameters.
        adjoint0 (jnp.array): Initial condition for the adjoint state.
        t_span (jnp.array): Time span for integration.
        potentials (list[callable]): List of potential functions.
        constraint (callable, optional): Constraint function to parametrize the coordinates.

    Returns:
        jnp.ndarray: Adjoint state trajectory over time.
    """

    
    @jax.jit 
    def augmented_dynamics(t, state):
        return jnp.array([dynamics(t, state[:3]), adjoint_dynamics(t, state[3:], state[:3], state[3:])])

    augmented_state = jnp.concatenate(state, jnp.zeros_like(q0))
    result = solve_ivp(augmented_dynamics, (t0, t), augmented_state)



    q, qdot = evolve_lagrangian(t0, t, q0, qdot0, mass, **kwargs)
    mass = jnp.ones(q.shape[1]) * mass
    q = jnp.flip(q, axis=1)
    qdot = jnp.flip(qdot, axis=1)
    print("mass", mass.shape)
    print("q", q.shape)
    print("qdot", qdot.shape)

    @jax.jit
    def adjoint_dynamics(adjoint_state, t, q, qdot):
        """Compute the equations of motion for the adjoint state."""
        df_dqdot = jax.jacrev(lagrangian_eom, 1)(q, qdot, mass, **kwargs)
        return -jnp.dot(adjoint_state, df_dqdot.T)

    lam = jnp.zeros(3)
    print(adjoint_dynamics(lam, 0.0, q[:, 0], qdot[:, 0]).shape)
        
    #     df = jnp.hstack([jnp.zeros_like(df_dqdot), df_dqdot])
    #     return -jnp.dot(adjoint_state, df)
    
    # # Use odeint to integrate the adjoint equations backward in time
    # adjoint_trajectory = odeint(dynamics, jnp.concatenate([jnp.zeros_like(q[0, :]), jnp.zeros_like(q[0, :])]), jnp.linspace(t, t0, n))



# print(evolve_lagrangian_adjoints(1.0, 0.0, q0=jnp.zeros(3), qdot0=jnp.array([0.5, -0.5, 0.1]), mass=jnp.ones(1), potentials=[potential(gravity, g=9.81)]))


@jax.jit
def loss(params, batch, time_step=None):
  state, targets = batch
  preds = jax.vmap(lagrangian_param_dynamics)(state)
  return jnp.mean((preds - targets) ** 2)