"""Validate the config-driven dual-energy port on the refactor branch.

Runs a quasi-1-D ideal-gas MHD problem (B=0) through the *real* high-level
``time_integration`` entry point with ``config.dual_energy`` False vs True:

  * non-dual returns an 11-var state and runs (regression: the g=None code path
    is unchanged);
  * dual returns a 12-var state (g appended as the last variable via
    ``registered_variables``), stays finite, and the carried g equals
    p/(gamma-1) at the end (the re-sync invariant);
  * the shared fields (rho, v, p, B) of the dual and non-dual runs agree to
    float rounding on an ordinary (no-cancellation) problem -> dual energy is
    no-harm.
"""
# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig, SimulationParams, finalize_config,
    get_registered_variables, construct_primitive_state, time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, StaticFloatVector,
)

GAMMA = 5.0 / 3.0
N = 32
NT = 8


def make(dual):
    per = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=True,
        dimensionality=3, backend=PALLAS,
        box_size=StaticFloatVector(1.0, NT / N, NT / N),
        num_cells=StaticFloatVector(N, NT, NT),
        dual_energy=dual,
        boundary_settings=BoundarySettings(per, per, per),
        return_snapshots=False,
    )


def run(dual, t_end=0.04):
    config = make(dual)
    rv = get_registered_variables(config)
    x = (np.arange(N) + 0.5) / N
    rho = np.where((x > 0.25) & (x < 0.75), 1.0, 0.125).astype(np.float32)
    p = np.where((x > 0.25) & (x < 0.75), 1.0, 0.1).astype(np.float32)
    f = lambda a: jnp.asarray(np.broadcast_to(a[:, None, None], (N, NT, NT)).copy())
    state = construct_primitive_state(
        config, rv, density=f(rho),
        velocity_x=f(np.zeros(N, np.float32)),
        velocity_y=f(np.zeros(N, np.float32)),
        velocity_z=f(np.zeros(N, np.float32)),
        magnetic_field_x=f(np.zeros(N, np.float32)),
        magnetic_field_y=f(np.full(N, 0.05, np.float32)),
        magnetic_field_z=f(np.zeros(N, np.float32)),
        gas_pressure=f(p),
    )
    params = SimulationParams(C_cfl=0.4, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-12, minimum_density=1e-10)
    config = finalize_config(config, state.shape)
    out = time_integration(state, config, params, rv)
    return np.asarray(out), rv


if __name__ == "__main__":
    off, rv0 = run(dual=False)
    on, rv1 = run(dual=True)
    print(f"[refactor dual-energy] non-dual: shape={off.shape} finite={np.all(np.isfinite(off))} (num_vars={rv0.num_vars})")
    print(f"[refactor dual-energy] dual    : shape={on.shape} finite={np.all(np.isfinite(on))} (num_vars={rv1.num_vars}, g_index={rv1.internal_energy_index})")
    pin = rv1.pressure_index
    g = on[rv1.internal_energy_index]
    g_expected = on[pin] / (GAMMA - 1.0)
    print(f"[refactor dual-energy] g == p/(gamma-1) ? max|g-p/(g-1)|={np.max(np.abs(g - g_expected)):.2e}")
    d = np.max(np.abs(off[: rv0.num_vars] - on[: rv0.num_vars]))
    print(f"[refactor dual-energy] no-harm: max|dual - nondual| over shared vars = {d:.2e} "
          f"rel={d / (np.abs(off).max() + 1e-30):.2e}")
