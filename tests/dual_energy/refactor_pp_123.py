"""PP-flux limiter on the Zhang-Shu 123 double-rarefaction problem.

Two streams move apart (u = -+ V) from uniform density/pressure; the centre
rarefies to very low (but physically POSITIVE) density. High-order WENO
undershoots density to negative there and crashes; the PP flux limiter keeps it
positive. This is the canonical case PP limiting is designed for (a *numerical*
undershoot, distinct from the physical hypersonic vacuum that no blend limiter
can fix).
"""
# ruff: noqa: E402
import numpy as np
from autocvd import autocvd
autocvd(num_gpus=1)

import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams, finalize_config,
    get_registered_variables, construct_primitive_state, time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS, PERIODIC_BOUNDARY, POSITIVITY_NONE,
    BoundarySettings, BoundarySettings1D, StaticFloatVector,
)

GAMMA = 5.0 / 3.0


def run(pp, V=4.0, p0=0.4, N=128, NT=8, t_end=0.1):
    per = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=True,
        dimensionality=3, backend=PALLAS, box_size=StaticFloatVector(1.0, NT / N, NT / N),
        num_cells=StaticFloatVector(N, NT, NT), positivity_preserving_flux=pp,
        # isolate the flux limiter: no per-stage/step floor at all
        positivity_per_stage_mode=POSITIVITY_NONE, positivity_per_step_mode=POSITIVITY_NONE,
        enforce_positivity=False,
        boundary_settings=BoundarySettings(per, per, per), return_snapshots=False,
    )
    rv = get_registered_variables(config)
    x = (np.arange(N) + 0.5) / N
    vx = np.where(x < 0.5, -V, V).astype(np.float32)     # two streams moving apart
    f = lambda a: jnp.asarray(np.broadcast_to(a[:, None, None], (N, NT, NT)).copy())
    z = np.zeros(N, np.float32)
    state = construct_primitive_state(
        config, rv, density=f(np.ones(N, np.float32)),
        velocity_x=f(vx), velocity_y=f(z), velocity_z=f(z),
        magnetic_field_x=f(z), magnetic_field_y=f(z), magnetic_field_z=f(z),
        gas_pressure=f(np.full(N, p0, np.float32)),
    )
    params = SimulationParams(C_cfl=0.3, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-12, minimum_density=1e-12)
    config = finalize_config(config, state.shape)
    out = np.asarray(time_integration(state, config, params, rv))
    di, pin = rv.density_index, rv.pressure_index
    return dict(finite=bool(np.all(np.isfinite(out))),
                rho_min=float(np.nanmin(out[di])), p_min=float(np.nanmin(out[pin])))


if __name__ == "__main__":
    cs = np.sqrt(GAMMA * 0.4)
    print("[123 problem] no positivity floor (isolating the flux limiter); "
          "OFF crashing while ON stays finite would justify PP-flux")
    for V in (6.0, 8.0, 10.0, 12.0):
        off = run(False, V=V)
        on = run(True, V=V)
        print(f"  V={V:4.1f} (M={V/cs:4.1f}): "
              f"OFF finite={off['finite']!s:5} rho_min={off['rho_min']:.2e} | "
              f"ON finite={on['finite']!s:5} rho_min={on['rho_min']:.2e}")
