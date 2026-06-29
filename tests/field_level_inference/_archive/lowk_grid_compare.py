#!/usr/bin/env python
"""Large-scale (low-k) inference: 32³ vs 64³ optimisation, evaluated at 64³.

Tests the principled multigrid claim: large-scale modes (|k| < n_lo/2, well below
both grids' Nyquist) are coherent on 32³ and 64³, so optimising them on the CHEAP
32³ grid should transfer to 64³ -- unlike the earlier proxy which optimised the
full 32³ band (incl. near-Nyquist jitter) and failed.

Consistent backgrounds: generate turb at 64³ once, restrict to 32³.  Optimise the
SAME band on each grid, then evaluate BOTH controls at 64³ (prolong the band-limited
velocity, 64³ background, forward) -> logo corr + temporal FWHM.  Reports the
per-step cost so we can quantify the saving.  PALLAS BACKWARDS + cache.  astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n-lo", type=int, default=16, help="large-scale band: keep |k| < n_lo/2.")
ap.add_argument("--contrast", type=float, default=0.01)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
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
NLO = args.n_lo


def make_cfg(N, forcing, backward):
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else FORWARDS,
        num_checkpoints=args.num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_use_triton=True)


def build(cfg, rv, rho, v, p, B):
    bxb, byb, bzb = initialize_interface_fields(B[0], B[1], B[2])
    return construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
        velocity_x=v[0], velocity_y=v[1], velocity_z=v[2],
        magnetic_field_x=B[0], magnetic_field_y=B[1], magnetic_field_z=B[2],
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb, interface_magnetic_field_z=bzb)


def load_image(path, N, contrast):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * contrast


def corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def fields(turb, rv, N):
    di, pi, vi, mi = rv.density_index, rv.pressure_index, rv.velocity_index, rv.magnetic_index
    def rs(x):
        return x if x.shape[-1] == N else (restrict(x, N) if N < x.shape[-1] else prolong(x, N))
    rho = rs(turb[di][None])[0]; p = rs(turb[pi][None])[0]
    v = rs(jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]]))
    B = rs(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]]))
    return rho, p, v, B


def optimise(N, turb, rv, t_end, params):
    """Optimise the |k|<NLO/2 velocity on the N³ grid; return the band-limited
    velocity (N³) and the mean per-step time."""
    cfg = finalize_config(make_cfg(N, False, True), (turb.shape[0],) + (N,) * 3)
    rvN = get_registered_variables(cfg); di, vi = rvN.density_index, rvN.velocity_index
    rho, p, v0, B = fields(turb, rv, N)
    tgt = load_image("logo.png", N, args.contrast); tgt = tgt / jnp.sum(tgt) * jnp.sum(rho)
    inv = params._replace(t_end=t_end)

    def lowpass(x):
        return prolong(restrict(x, NLO), N)

    def loss(theta):
        vv = lowpass(theta)
        s = build(cfg, rvN, rho, (vv[0], vv[1], vv[2]), p, (B[0], B[1], B[2]))
        proj = jnp.sum(time_integration(s, cfg, inv, rvN)[di], axis=2)
        return jnp.mean((proj - tgt) ** 2)

    theta = v0
    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
    m = jnp.zeros_like(theta); vv_ = jnp.zeros_like(theta)

    @jax.jit
    def step(theta, m, vv_, g, t):
        m = b1 * m + (1 - b1) * g; vv_ = b2 * vv_ + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(vv_ / (1 - b2 ** t)) + eps)
        return theta, m, vv_

    best = (float(loss(theta)), theta); times = []
    for s in range(args.num_steps):
        ts = time.time()
        l, g = vg(theta); l = float(l)
        if l < best[0]:
            best = (l, theta)
        theta, m, vv_ = step(theta, m, vv_, g, float(s + 1))
        dt = time.time() - ts
        if s > 0:
            times.append(dt)
        if s < 2 or s % 5 == 0 or s == args.num_steps - 1:
            print(f"   [{N}³] step {s:3d}: L={l:.4e} ({dt:.1f}s)", flush=True)
    return lowpass(best[1]), (np.mean(times) if times else 0.0), best[0]


def eval_at_64(turb, rv, v_band_src_N, t_end, params):
    """Put the band-limited velocity (prolonged to 64³) on the 64³ background,
    forward-evaluate, and scan corr-vs-time -> (peak corr, FWHM%)."""
    N = 64
    cfg = finalize_config(make_cfg(N, False, False), turb.shape)
    rho, p, _, B = fields(turb, rv, N)
    di = rv.density_index
    v = v_band_src_N if v_band_src_N.shape[-1] == N else prolong(v_band_src_N, N)
    tgt = load_image("logo.png", N, args.contrast); tgt = tgt / jnp.sum(tgt) * jnp.sum(rho)
    state = build(cfg, rv, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
    ts = np.linspace(0.6 * t_end, 1.6 * t_end, 31)
    cs = np.array([corr(jnp.sum(time_integration(state, cfg, params._replace(t_end=float(t)), rv)[di], axis=2), tgt)
                   for t in ts])
    pk = int(np.argmax(cs)); half = cs[pk] / 2
    ab = ts[cs >= half]; fwhm = (ab.max() - ab.min()) if ab.size > 1 else 0.0
    return cs[pk], ts[pk], 100 * fwhm / t_end


def main():
    print(f"=== low-k 32 vs 64 compare  band|k|<{NLO//2}  contrast={args.contrast} "
          f"steps={args.num_steps} ===", flush=True)
    cfgg = make_cfg(64, True, False)
    rv = get_registered_variables(cfgg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    sh = (64,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    vi = rv.velocity_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    print(f"turb gen done  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    print("--- optimise on 32³ (cheap) ---", flush=True)
    v32, dt32, L32 = optimise(32, turb, rv, t_end, params)
    print("--- optimise on 64³ (reference) ---", flush=True)
    v64, dt64, L64 = optimise(64, turb, rv, t_end, params)

    c32, t32, w32 = eval_at_64(turb, rv, v32, t_end, params)
    c64, t64, w64 = eval_at_64(turb, rv, v64, t_end, params)
    print("\n==== RESULT (both evaluated at 64³) ====", flush=True)
    print(f" 32³-opt: bestL={L32:.3e}  ~{dt32:.1f}s/step  ->64³ peakcorr={c32:.3f} "
          f"@t={t32:.3f}  FWHM={w32:.1f}%", flush=True)
    print(f" 64³-opt: bestL={L64:.3e}  ~{dt64:.1f}s/step  ->64³ peakcorr={c64:.3f} "
          f"@t={t64:.3f}  FWHM={w64:.1f}%", flush=True)
    print(f" per-step speedup 64/32 = {dt64/max(dt32,1e-9):.1f}x", flush=True)
    np.save("v_band_lowk_32opt.npy", np.asarray(v32))
    np.save("v_band_lowk_64opt.npy", np.asarray(v64))


if __name__ == "__main__":
    main()
