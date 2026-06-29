#!/usr/bin/env python
"""k-space expansion (coarse-to-fine in k) to recover peak fidelity from a robust
low-k base, comparing two warm starts.

Band-limit the velocity with a differentiable k-space MASK (a traced array, so
widening the band does NOT recompile the Pallas kernel): v = irfft(rfft(theta)*mask).
Start at a low k_cut and progressively widen it, optimising at each stage; the
large-scale base stays fixed while finer modes are added to sharpen the logo.

Two warm starts, same schedule:
  A '32': the 32³-optimised large-scale velocity (prolongation-free multigrid warm
          start; loaded from v_band_lowk_32opt.npy) -- pre-converged cheaply.
  B 'turb': the raw turbulent velocity ('now' warm start).
Reports corr after each band + final temporal FWHM for both.  PALLAS BACKWARDS.
astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--bands", type=str, default="8:6,16:8,32:16",
                help="k_cut:steps schedule (k_cut in integer wavenumbers; 32 = full at 64³).")
ap.add_argument("--warm32", type=str, default="v_band_lowk_32opt.npy")
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
args = ap.parse_args()

SCHED = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in args.bands.split(",")]

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
from _spectral_ops import prolong

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
B0 = (13.5 * u.microgauss / c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT = (4.3e34 * u.erg / u.s).to(CU.code_energy / CU.code_time).value
N = 64


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
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_use_triton=True)


def build(cfg, rv, rho, v, p, B):
    bxb, byb, bzb = initialize_interface_fields(B[0], B[1], B[2])
    return construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
        velocity_x=v[0], velocity_y=v[1], velocity_z=v[2],
        magnetic_field_x=B[0], magnetic_field_y=B[1], magnetic_field_z=B[2],
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb, interface_magnetic_field_z=bzb)


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def kmask(kcut):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    return jnp.asarray((kk < kcut).astype(np.float64))


def main():
    print(f"=== k-expansion  sched={SCHED}  warm32={args.warm32} ===", flush=True)
    cfgg = make_cfg(True, False); rv = get_registered_variables(cfgg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    sh = (N,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    di, vi = rv.density_index, rv.velocity_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    print(f"turb gen done  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    cfg = finalize_config(make_cfg(False, True), turb.shape)
    cfg_f = finalize_config(make_cfg(False, False), turb.shape)
    inv = params._replace(t_end=t_end)
    tgt = load_image("logo.png", N); tgt = tgt / jnp.sum(tgt) * jnp.sum(turb[di])
    rho_bg, p_bg = turb[di], turb[rv.pressure_index]
    Bbg = (turb[rv.magnetic_index.x], turb[rv.magnetic_index.y], turb[rv.magnetic_index.z])

    def loss(theta, mask):
        v = jnp.fft.ifftn(jnp.fft.fftn(theta, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real
        s = build(cfg, rv, rho_bg, (v[0], v[1], v[2]), p_bg, Bbg)
        proj = jnp.sum(time_integration(s, cfg, inv, rv)[di], axis=2)
        return jnp.mean((proj - tgt) ** 2)

    def proj_at(theta, mask, cfgx, t):
        v = jnp.fft.ifftn(jnp.fft.fftn(theta, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real
        s = build(cfgx, rv, rho_bg, (v[0], v[1], v[2]), p_bg, Bbg)
        return jnp.sum(time_integration(s, cfgx, params._replace(t_end=float(t)), rv)[di], axis=2)

    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr

    def kexpand(theta0, label):
        theta = theta0
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)
        tstep = 0
        for (kcut, steps) in SCHED:
            mask = kmask(kcut)
            for s in range(steps):
                t0 = time.time()
                l, g = vg(theta, mask)
                # Adam (continuous t across bands)
                tstep += 1
                m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
                theta = theta - lr * (m / (1 - b1 ** tstep)) / (jnp.sqrt(vv / (1 - b2 ** tstep)) + eps)
                if s == steps - 1:
                    cval = corr(proj_at(theta, mask, cfg_f, t_end), tgt)
                    print(f"   [{label}] kcut={kcut:2d} step {tstep:3d}: L={float(l):.4e} "
                          f"corr={cval:.4f} ({time.time()-t0:.1f}s)", flush=True)
        # final temporal-width scan with the full final mask
        mask = kmask(SCHED[-1][0])
        ts = np.linspace(0.6 * t_end, 1.6 * t_end, 25)
        cs = np.array([corr(proj_at(theta, mask, cfg_f, t), tgt) for t in ts])
        pk = int(np.argmax(cs)); half = cs[pk] / 2
        ab = ts[cs >= half]; fwhm = (ab.max() - ab.min()) if ab.size > 1 else 0.0
        print(f"   [{label}] FINAL peak corr={cs[pk]:.4f} @t={ts[pk]:.3f}  "
              f"FWHM={100*fwhm/t_end:.1f}%", flush=True)
        return theta, cs[pk], 100 * fwhm / t_end

    # warm A: 32³-optimised large-scale velocity (prolongation-free multigrid)
    v32 = jnp.asarray(np.load(args.warm32))
    theta_A = prolong(v32, N) if v32.shape[-1] != N else v32
    # warm B: raw turbulent velocity ('now')
    theta_B = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])

    print("--- WARM A: 32³ multigrid warm start ---", flush=True)
    _, cA, wA = kexpand(theta_A, "A:32mg")
    print("--- WARM B: turbulent ('now') warm start ---", flush=True)
    _, cB, wB = kexpand(theta_B, "B:turb")
    print(f"\n==== k-EXPANSION RESULT ====", flush=True)
    print(f" A 32³-warm : final corr={cA:.3f}  FWHM={wA:.1f}%", flush=True)
    print(f" B turb-warm: final corr={cB:.3f}  FWHM={wB:.1f}%", flush=True)


if __name__ == "__main__":
    main()
