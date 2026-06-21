"""Where do dual energy and PP-flux limiting actually help (refactor)?

(A) Dual-energy isolated win: a high-Mach *uniform* flow (no rarefaction, density
    stays ~constant) carrying a small pressure bump, with e_int/E < float32 eps.
    Only the pressure cancellation matters here (no vacuum), so this isolates the
    dual-energy switch from any positivity effect.

(B) Robustness matrix on M~20 3-D decaying turbulence with the *correct*
    vacuum-robust positivity (REDISTRIBUTE conserves momentum AND applies the
    velocity cap; HARD_FLOOR ignores the cap). Compares plain / +cap / +dual /
    +dual+PP to see which mechanism buys survival.
"""
# ruff: noqa: E402
import sys
import numpy as np

from autocvd import autocvd
autocvd(num_gpus=1)

import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams, finalize_config,
    get_registered_variables, construct_primitive_state, time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS, PERIODIC_BOUNDARY,
    POSITIVITY_HARD_FLOOR, POSITIVITY_REDISTRIBUTE,
    BoundarySettings, BoundarySettings1D, StaticFloatVector,
)

GAMMA = 5.0 / 3.0


def _per():
    return BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)


# ----------------------------------------------------------------------------
# (A) dual-energy isolated win — high-Mach uniform flow + pressure bump
# ----------------------------------------------------------------------------
def run_uniform(dual, V=50.0, p0=1e-5, N=64, NT=8, t_end=None):
    per = _per()
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=True,
        dimensionality=3, backend=PALLAS, box_size=StaticFloatVector(1.0, NT / N, NT / N),
        num_cells=StaticFloatVector(N, NT, NT), dual_energy=dual,
        positivity_per_stage_mode=POSITIVITY_HARD_FLOOR,
        positivity_per_step_mode=POSITIVITY_HARD_FLOOR,
        boundary_settings=BoundarySettings(per, per, per), return_snapshots=False,
    )
    rv = get_registered_variables(config)
    x = (np.arange(N) + 0.5) / N
    bump = 0.5 * p0 * np.exp(-((x - 0.5) ** 2) / (0.02 ** 2))
    p = (p0 + bump).astype(np.float32)
    f = lambda a: jnp.asarray(np.broadcast_to(a[:, None, None], (N, NT, NT)).copy())
    z = np.zeros(N, np.float32)
    state = construct_primitive_state(
        config, rv, density=f(np.ones(N, np.float32)),
        velocity_x=f(np.full(N, V, np.float32)), velocity_y=f(z), velocity_z=f(z),
        magnetic_field_x=f(z), magnetic_field_y=f(np.full(N, 0.05, np.float32)), magnetic_field_z=f(z),
        gas_pressure=f(p),
    )
    if t_end is None:
        t_end = 0.4 / V
    params = SimulationParams(C_cfl=0.4, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-12, minimum_density=1e-10)
    config = finalize_config(config, state.shape)
    out = np.asarray(time_integration(state, config, params, rv))
    pin = rv.pressure_index
    return dict(finite=bool(np.all(np.isfinite(out))),
                bump_amp=float(np.nanmax(out[pin]) - np.nanmin(out[pin])),
                p_min=float(np.nanmin(out[pin])))


# ----------------------------------------------------------------------------
# (B) M~20 turbulence robustness matrix
# ----------------------------------------------------------------------------
def solenoidal(N, vrms, seed=1, kmax=4):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2; K2[0, 0, 0] = 1.0
    band = ((np.sqrt(K2) <= kmax) & (np.sqrt(K2) > 0)).astype(np.float64)
    f = [(rng.normal(size=(N,)*3) + 1j * rng.normal(size=(N,)*3)) * band for _ in range(3)]
    kdotf = (KX * f[0] + KY * f[1] + KZ * f[2]) / K2
    f = [f[i] - (KX, KY, KZ)[i] * kdotf for i in range(3)]
    v = np.stack([np.fft.ifftn(fi).real for fi in f])
    v *= vrms / np.sqrt(np.mean(v[0]**2 + v[1]**2 + v[2]**2))
    return v.astype(np.float32)


def run_turb(M, dual, stage_mode, cap, N=48, p0=1e-3, bguide=0.05, t_end=0.15):
    per = _per()
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=True,
        dimensionality=3, backend=PALLAS, box_size=StaticFloatVector(1.0, 1.0, 1.0),
        num_cells=StaticFloatVector(N, N, N), dual_energy=dual,
        positivity_per_stage_mode=stage_mode, positivity_per_step_mode=stage_mode,
        boundary_settings=BoundarySettings(per, per, per), return_snapshots=False,
    )
    rv = get_registered_variables(config)
    cs = np.sqrt(GAMMA * p0); vrms = M * cs
    v = solenoidal(N, vrms)
    g = lambda a: jnp.asarray(a)
    z = np.zeros((N, N, N), np.float32)
    state = construct_primitive_state(
        config, rv, density=g(np.ones((N, N, N), np.float32)),
        velocity_x=g(v[0]), velocity_y=g(v[1]), velocity_z=g(v[2]),
        magnetic_field_x=g(z), magnetic_field_y=g(np.full((N, N, N), bguide, np.float32)), magnetic_field_z=g(z),
        gas_pressure=g(np.full((N, N, N), p0, np.float32)),
    )
    params = SimulationParams(C_cfl=0.3, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-12, minimum_density=1e-10,
                              positivity_max_velocity=cap)
    config = finalize_config(config, state.shape)
    out = np.asarray(time_integration(state, config, params, rv))
    return dict(finite=bool(np.all(np.isfinite(out))),
                rho_min=float(np.nanmin(out[rv.density_index])),
                p_min=float(np.nanmin(out[rv.pressure_index])),
                vmax=float(np.nanmax(np.abs(out[rv.velocity_index.x]))))


if __name__ == "__main__":
    print("=== (A) dual-energy isolated win: uniform V=50 flow, p0=1e-5 (e_int/E~%.0e) ==="
          % ((1e-5 / (GAMMA - 1)) / (0.5 * 50.0**2)))
    for dual in (False, True):
        r = run_uniform(dual)
        print(f"  dual={dual!s:5}: finite={r['finite']} bump_amp={r['bump_amp']:.2e} p_min={r['p_min']:.2e}")

    M = 20.0
    cs = np.sqrt(GAMMA * 1e-3); vcap = 5.0 * M * cs
    print(f"\n=== (B) M={M} 3-D turbulence robustness (vcap={vcap:.2f}) ===")
    matrix = [
        ("plain (HARD_FLOOR, no cap)",  dict(dual=False, stage_mode=POSITIVITY_HARD_FLOOR, cap=jnp.inf)),
        ("REDISTRIBUTE + vcap",         dict(dual=False, stage_mode=POSITIVITY_REDISTRIBUTE, cap=vcap)),
        ("+ dual energy",               dict(dual=True,  stage_mode=POSITIVITY_REDISTRIBUTE, cap=vcap)),
    ]
    for label, kw in matrix:
        r = run_turb(M, **kw)
        print(f"  {label:34}: finite={r['finite']} rho_min={r['rho_min']:.2e} "
              f"p_min={r['p_min']:.2e} max|vx|={r['vmax']:.3e}")
