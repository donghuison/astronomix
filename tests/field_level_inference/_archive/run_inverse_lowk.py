#!/usr/bin/env python
"""Low-k (coarse-band) velocity inference -- 'prolongation-free multigrid'.

Parameterise the initial velocity at full resolution but pass it through a
DIFFERENTIABLE spectral low-pass before the sim: v = lowpass_{|k|<n_lo/2}(theta).
Autodiff then only updates the low-k band (high-k gradient is killed by the
mask), so we optimise the coarse modes with no explicit coarse grid /
prolongation.  Hypothesis: a low-k velocity advects coherently, so the logo
forms over a LONGER time window (not the sharp transient of the full-mode
solution), at the cost of a blurrier peak.

After optimisation we scan corr-vs-time to measure the temporal width and
compare to the full-mode solution.  PALLAS BACKWARDS + compile cache.  astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=64)
ap.add_argument("--n-lo", type=int, default=32, help="low-pass band: keep |k| < n_lo/2.")
ap.add_argument("--contrast", type=float, default=0.01,
                help="target logo overdensity above the mean (0.01 = 1%, the original; "
                     "larger -> stronger bulk movement).")
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out", type=str, default=None)
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
import numpy as np
import jax
_CACHE = os.path.expanduser("~/.cache/astronomix_jax_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 30)
import jax.numpy as jnp
from PIL import Image

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, FORWARDS, finalize_config,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astropy import units as u
import astropy.constants as c
from _spectral_ops import prolong, restrict

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
B0 = (13.5 * u.microgauss / c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT = (4.3e34 * u.erg / u.s).to(CU.code_energy / CU.code_time).value


def load_image(path, N, contrast=0.01):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * contrast


def corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def main():
    N, nlo = args.resolution, args.n_lo
    out = args.out or f"best_state_lowk{nlo}_{N}.npy"

    def make_cfg(forcing, backward):
        return SimulationConfig(
            solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
            enforce_positivity=True, donate_state=False, dimensionality=3,
            box_size=1.0, num_cells=N,
            differentiation_mode=BACKWARDS if backward else FORWARDS,
            num_checkpoints=args.num_checkpoints,
            turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
            boundary_settings=BoundarySettings(
                *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
            backend=PALLAS, pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True, pallas_interpret=False)

    cfg = make_cfg(forcing=True, backward=False)
    rv = get_registered_variables(cfg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    sh = (N,) * 3; z = jnp.zeros(sh)
    bxb, byb, bzb = initialize_interface_fields(jnp.ones(sh) * B0, z, z)
    s0 = construct_primitive_state(
        config=cfg, registered_variables=rv, density=jnp.ones(sh) * RHO0,
        velocity_x=z, velocity_y=z, velocity_z=z, gas_pressure=jnp.ones(sh) * P0,
        magnetic_field_x=jnp.ones(sh) * B0, magnetic_field_y=z, magnetic_field_z=z,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb, interface_magnetic_field_z=bzb)
    cfg = finalize_config(cfg, s0.shape)
    t_gen = (24 * 1e4 * u.yr).to(CU.code_time).value
    turb = jax.block_until_ready(time_integration(s0, cfg, params._replace(t_end=t_gen), rv))
    di, vi = rv.density_index, rv.velocity_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    print(f"=== low-k inference  N={N} n_lo={nlo}  rms={rms:.4f}  t_end={t_end:.4f} ===", flush=True)

    cfg = finalize_config(make_cfg(forcing=False, backward=True), turb.shape)
    inv = params._replace(t_end=t_end)
    tgt = load_image("logo.png", N, args.contrast); tgt = tgt / jnp.sum(tgt) * jnp.sum(turb[di])

    def lowpass(x):                       # keep |k| < nlo/2 (prolongation-free)
        return prolong(restrict(x, nlo), N)

    def loss(theta):
        v = lowpass(theta)
        s = turb.at[vi.x].set(v[0]).at[vi.y].set(v[1]).at[vi.z].set(v[2])
        proj = jnp.sum(time_integration(s, cfg, inv, rv)[di], axis=2)
        return jnp.mean((proj - tgt) ** 2)

    theta = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])  # high-k part is inert
    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
    m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)

    @jax.jit
    def step(theta, m, vv, g, t):
        m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + eps)
        return theta, m, vv

    best = (float(loss(theta)), theta)
    print(f"start L={best[0]:.4e}", flush=True)
    for s in range(args.num_steps):
        ts = time.time()
        l, g = vg(theta); l = float(l)
        if l < best[0]:
            best = (l, theta)
        theta, m, vv = step(theta, m, vv, g, float(s + 1))
        if s < 2 or s % 5 == 0 or s == args.num_steps - 1:
            print(f"   step {s:3d}: L={l:.4e}  ({time.time()-ts:.1f}s)", flush=True)
    theta = best[1]
    v = lowpass(theta)
    best_state = turb.at[vi.x].set(v[0]).at[vi.y].set(v[1]).at[vi.z].set(v[2])
    np.save(out, np.asarray(best_state))
    print(f"optimised bestL={best[0]:.4e}; wrote {out}", flush=True)

    # --- temporal-width scan: how long does the logo persist? ---
    cfg_f = finalize_config(make_cfg(forcing=False, backward=False), turb.shape)
    ts_scan = np.linspace(0.6 * t_end, 1.6 * t_end, 31)
    cs = []
    for t in ts_scan:
        fin = time_integration(best_state, cfg_f, params._replace(t_end=float(t)), rv)
        cs.append(corr(jnp.sum(fin[di], axis=2), tgt))
    cs = np.array(cs)
    np.savez(f"lowk{nlo}_{N}_timescan.npz", ts=ts_scan, corr=cs, t_end=t_end)
    pk = int(np.argmax(cs))
    half = cs[pk] / 2
    above = ts_scan[cs >= half]
    fwhm = (above.max() - above.min()) if above.size > 1 else 0.0
    print(f"TIME-WIDTH: peak corr={cs[pk]:.4f} at t={ts_scan[pk]:.4f}  "
          f"FWHM={fwhm:.4f} ({100*fwhm/t_end:.1f}% of t_end)", flush=True)


if __name__ == "__main__":
    main()
