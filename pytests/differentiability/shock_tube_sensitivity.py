"""
AD vs finite-difference gradients through the Sod shock tube.

Parameterizes the Sod initial condition by the left/right density, pressure and
velocity, integrates it through the finite-difference solver with a fixed
timestep, and differentiates a quadratic cost of the final state with respect to
those six parameters. The reverse-mode AD gradient is compared against a central
finite-difference baseline and the per-parameter gradients are plotted, checking
that the solver is differentiable through shocks.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    OPEN_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BoundarySettings1D,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config


# Figures are written to a local figures/ directory next to this script.
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# 64-bit precision so the AD gradient and the central finite-difference baseline
# agree to well below the step-size truncation error.
jax.config.update("jax_enable_x64", True)


def make_config_and_params(num_cells, L, t_end, num_timesteps):
    """
    Build a 1D open-boundary config with a fixed timestep and its parameters.

    The timestep is fixed (rather than CFL-adaptive) so that the cost J(theta) is
    a smooth function of the parameters — an adaptive step count would introduce
    kinks at CFL transitions that break the finite-difference comparison.

    Args:
        num_cells: The number of grid cells along the single spatial dimension.
        L: The box size.
        t_end: The final integration time.
        num_timesteps: The fixed number of timesteps to take.

    Returns:
        A tuple (config, params) of the simulation configuration and parameters.
    """
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        progress_bar=False,
        differentiation_mode=BACKWARDS,
        backend=NATIVE_JAX,
        mhd=False,
        dimensionality=1,
        box_size=L,
        num_cells=num_cells,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY,
            right_boundary=OPEN_BOUNDARY,
        ),
        return_snapshots=False,
        fixed_timestep=True,
        num_timesteps=num_timesteps,
    )
    params = SimulationParams(
        C_cfl=0.4,
        t_end=t_end,
        gamma=5 / 3,
        gravitational_constant=0.0,
    )
    return config, params


def build_initial_state(theta, config, registered_variables, helper_data, x_split):
    """
    Build the Sod initial condition parameterized by theta.

    The parameter vector is theta = (rho_L, p_L, u_L, rho_R, p_R, u_R). The state
    is exactly linear in theta because each per-cell field is a jnp.where that
    simply selects one theta component depending on which side of x_split the
    cell lies on.

    Args:
        theta: The six-component parameter vector (left then right state).
        config: The simulation configuration.
        registered_variables: The registered variables.
        helper_data: The helper data providing the geometric cell centers.
        x_split: The x-coordinate of the diaphragm separating the two states.

    Returns:
        The primitive initial state array.
    """
    rho_L, p_L, u_L, rho_R, p_R, u_R = theta
    x = jnp.squeeze(helper_data.geometric_centers)
    left = x < x_split
    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)
    p = jnp.where(left, p_L, p_R)
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u,
        gas_pressure=p,
    )


def cost(theta, config, params, registered_variables, helper_data, x_split):
    """
    Quadratic cost of the final state for the Sod problem parameterized by theta.

    The cost is J(theta) = 0.5 * sum_x (rho^2 + v^2 + p^2) * dx, evaluated on the
    state obtained by integrating the theta-parameterized initial condition to
    the final time.

    Args:
        theta: The six-component parameter vector (left then right state).
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        helper_data: The helper data providing the geometric cell centers.
        x_split: The x-coordinate of the diaphragm separating the two states.

    Returns:
        The scalar cost J evaluated on the final state.
    """
    state0 = build_initial_state(theta, config, registered_variables, helper_data, x_split)
    config_final = finalize_config(config, state0.shape)
    state_T = time_integration(state0, config_final, params, registered_variables)
    rho = state_T[registered_variables.density_index]
    v = state_T[registered_variables.velocity_index]
    p = state_T[registered_variables.pressure_index]
    dx = config.box_size / config.num_cells
    return 0.5 * jnp.sum(rho**2 + v**2 + p**2) * dx


def fd_gradient(theta, h, jitted_cost):
    """
    Central finite-difference gradient baseline.

    Each component is estimated as g_i = (J(theta + h e_i) - J(theta - h e_i)) / 2h
    using two forward solver evaluations per parameter.

    Args:
        theta: The parameter vector to differentiate at.
        h: The finite-difference step size.
        jitted_cost: A compiled callable mapping theta to the scalar cost J.

    Returns:
        A numpy array holding the finite-difference gradient of J at theta.
    """
    g = np.zeros(theta.size)
    for i in range(theta.size):
        ei = jnp.zeros_like(theta).at[i].set(1.0)
        Jp = float(jitted_cost(theta + h * ei))
        Jm = float(jitted_cost(theta - h * ei))
        g[i] = (Jp - Jm) / (2.0 * h)
    return g


# -------------------------------------------------------------
# ======================= ↓ setup ↓ ===========================
# -------------------------------------------------------------

theta0 = jnp.array([1.0, 1.0, 0.0, 0.125, 0.1, 0.0])   # standard Sod state
param_names = ["rho_L", "p_L", "u_L", "rho_R", "p_R", "u_R"]
N, L, x_split = 200, 1.0, 0.5
# A fixed number of timesteps keeps the finite-difference theta-sweep CFL-stable.
t_end, num_timesteps = 0.2, 250
h = 1e-4

config, params = make_config_and_params(N, L, t_end, num_timesteps)
registered_variables = get_registered_variables(config)
helper_data = get_helper_data(config)

jitted_cost = jax.jit(
    lambda th: cost(th, config, params, registered_variables, helper_data, x_split)
)
jitted_ad = jax.jit(
    lambda th: jax.grad(
        lambda t: cost(t, config, params, registered_variables, helper_data, x_split)
    )(th)
)

# -------------------------------------------------------------
# ======================= ↑ setup ↑ ===========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ AD vs finite difference ↓ ================
# -------------------------------------------------------------

print(f"J(theta0) = {float(jitted_cost(theta0)):.6e}")
g_ad = np.asarray(jitted_ad(theta0))
g_fd = fd_gradient(theta0, h, jitted_cost)
for name, ga, gf in zip(param_names, g_ad, g_fd):
    print(f"  d J / d {name:<6s}  AD={ga: .6e}   FD={gf: .6e}")

rel = float(np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_ad) + 1e-30))
print(f"||g_AD - g_FD|| / ||g_AD|| = {rel:.3e}")

# -------------------------------------------------------------
# ================ ↑ AD vs finite difference ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============ ↓ plot per-parameter gradients ↓ ===============
# -------------------------------------------------------------

x_pos = np.arange(len(param_names))
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.scatter(x_pos, g_fd, s=120, marker="^", color="tab:orange",
           edgecolors="black", label=f"central FD (h={h:.0e})")
ax.scatter(x_pos, g_ad, s=180, marker="x", color="black", linewidths=2.5, label="AD")
ax.axhline(0, color="gray", linewidth=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, rotation=30)
ax.set_ylabel(r"$\partial J / \partial \theta_i$")
ax.set_title("Sod shock tube: AD vs finite-difference gradients")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "shock_tube_sensitivity.png", dpi=200)

# -------------------------------------------------------------
# ============ ↑ plot per-parameter gradients ↑ ===============
# -------------------------------------------------------------
