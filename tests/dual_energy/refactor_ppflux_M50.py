"""M~50 adiabatic-MHD stress test on the refactor branch.

3-D decaying high-Mach turbulence (cold gas, random solenoidal velocity) run
through the real ``time_integration`` with dual energy ON, comparing the
positivity-preserving flux limiter OFF vs ON. Without PP limiting the WENO
drives density to vacuum (v = mom/rho -> NaN); with PP limiting the density
stays >= floor and the run should remain finite.
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
    POSITIVITY_HARD_FLOOR, BoundarySettings, BoundarySettings1D, StaticFloatVector,
)

GAMMA = 5.0 / 3.0
N = 48
M = 50.0
P0 = 1e-3
BGUIDE = 0.05


def solenoidal(N, vrms, seed=1, kmax=4):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    band = ((np.sqrt(K2) <= kmax) & (np.sqrt(K2) > 0)).astype(np.float64)
    f = [(rng.normal(size=(N,)*3) + 1j * rng.normal(size=(N,)*3)) * band for _ in range(3)]
    kdotf = (KX * f[0] + KY * f[1] + KZ * f[2]) / K2
    f = [f[i] - (KX, KY, KZ)[i] * kdotf for i in range(3)]
    v = np.stack([np.fft.ifftn(fi).real for fi in f])
    v *= vrms / np.sqrt(np.mean(v[0]**2 + v[1]**2 + v[2]**2))
    return v.astype(np.float32)


def run(pp_flux, vmax_cap=jnp.inf, t_end=0.08):
    per = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=True,
        dimensionality=3, backend=PALLAS, box_size=StaticFloatVector(1.0, 1.0, 1.0),
        num_cells=StaticFloatVector(N, N, N),
        dual_energy=True, positivity_preserving_flux=pp_flux,
        positivity_per_stage_mode=POSITIVITY_HARD_FLOOR,
        positivity_per_step_mode=POSITIVITY_HARD_FLOOR,
        boundary_settings=BoundarySettings(per, per, per), return_snapshots=False,
    )
    rv = get_registered_variables(config)
    cs = np.sqrt(GAMMA * P0); vrms = M * cs
    v = solenoidal(N, vrms)
    g = lambda a: jnp.asarray(a)
    state = construct_primitive_state(
        config, rv, density=g(np.ones((N, N, N), np.float32)),
        velocity_x=g(v[0]), velocity_y=g(v[1]), velocity_z=g(v[2]),
        magnetic_field_x=g(np.zeros((N, N, N), np.float32)),
        magnetic_field_y=g(np.full((N, N, N), BGUIDE, np.float32)),
        magnetic_field_z=g(np.zeros((N, N, N), np.float32)),
        gas_pressure=g(np.full((N, N, N), P0, np.float32)),
    )
    params = SimulationParams(C_cfl=0.3, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-12, minimum_density=1e-10,
                              positivity_max_velocity=vmax_cap)
    config = finalize_config(config, state.shape)
    out = np.asarray(time_integration(state, config, params, rv))
    pin = rv.pressure_index
    return dict(finite=bool(np.all(np.isfinite(out))),
                rho_min=float(np.nanmin(out[rv.density_index])),
                p_min=float(np.nanmin(out[pin])),
                vmax=float(np.nanmax(np.abs(out[rv.velocity_index.x]))))


if __name__ == "__main__":
    cs = np.sqrt(GAMMA * P0)
    print(f"[M50 refactor] N={N} M={M} cs={cs:.3f} v_rms={M*cs:.3f} mean e_int/E~{(P0/(GAMMA-1))/(0.5*(M*cs)**2):.1e}")
    vcap = 5.0 * M * cs   # ~10: well above v_rms but bounds the vacuum spikes
    for pp, cap, tag in (
        (False, jnp.inf, "PP off, no cap "),
        (True, jnp.inf, "PP on,  no cap "),
        (False, vcap, "PP off, vcap   "),
        (True, vcap, "PP on,  vcap   "),
    ):
        r = run(pp, vmax_cap=cap)
        print(f"[M50 refactor] {tag}: finite={r['finite']} rho_min={r['rho_min']:.2e} "
              f"p_min={r['p_min']:.2e} max|vx|={r['vmax']:.3e}")
