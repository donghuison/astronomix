#!/usr/bin/env python
"""MONEY PLOT: terminal (logo-matching) loss vs optimisation wall-time for three
initial-velocity reconstruction strategies on the SAME logo target.  We accept the
inverse is degenerate (the recovered velocity is *a* state reproducing the logo, not
*the* true IC -- see identifiability.png); the well-posed, method-relevant metric is
the terminal loss = z-projection MSE vs the logo, at the fine grid (64³).

  naive      : optimise ALL Fourier modes at 64³ from step 1 (full mask).
  k-windowing: k-expansion schedule (8->16->32) entirely at 64³.
  multigrid  : SAME schedule, but low-k bands on the cheap 32³ grid (prolongation-free
               spectral transfer), high-k bands on 64³ -- our warm-start method.

All start from the SAME turbulent velocity ('now' warm start).  The multigrid's
32³ pre-optimisation time IS counted.  During the 32³ stage the plotted 64³ loss is a
forward-only diagnostic (prolong theta -> 64³ forward), NOT counted in wall-time.
naive & k-windowing share one compiled 64³ loss (mask is a traced arg).  PALLAS
BACKWARDS, astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--bands", type=str, default="8:10,16:10,32:14",
                help="k_cut:steps schedule (32 = full band at 64³).")
ap.add_argument("--coarse-cut", type=int, default=8,
                help="multigrid: bands with k_cut <= this run on 32³ (rest on 64³).")
ap.add_argument("--diag-every", type=int, default=2,
                help="evaluate the 64³ diagnostic loss every N steps during the 32³ stage.")
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out", type=str, default="bench_money.npz")
ap.add_argument("--only", choices=["naive", "k-windowing", "multigrid", "all"], default="all",
                help="run a single method (e.g. extend multigrid to equal wall-time).")
args = ap.parse_args()

SCHED = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in args.bands.split(",")]
TOTAL_STEPS = sum(s for _, s in SCHED)

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


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def kmask(N, kcut):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    return jnp.asarray((np.sqrt(KX**2 + KY**2 + KZ**2) < kcut).astype(np.float64))


def lowpass(theta, mask):
    return jnp.fft.ifftn(jnp.fft.fftn(theta, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real


def main():
    print(f"=== MONEY  bands={SCHED} (total {TOTAL_STEPS})  coarse_cut={args.coarse_cut} ===",
          flush=True)
    cfgg = make_cfg(64, True, False); rv = get_registered_variables(cfgg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    sh = (64,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    di, vi, pi, mi = rv.density_index, rv.velocity_index, rv.pressure_index, rv.magnetic_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    inv = params._replace(t_end=t_end)
    print(f"turb gen done  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    # per-grid backward configs, backgrounds, and logo targets (proj normalised to mass)
    cfgB = {N: finalize_config(make_cfg(N, False, True), (11,) + (N,) * 3) for N in (32, 64)}
    cfg64f = finalize_config(make_cfg(64, False, False), turb.shape)   # forward (diagnostic)
    rvN = {N: get_registered_variables(make_cfg(N, False, True)) for N in (32, 64)}

    def bg(N):
        rs = lambda x: x if x.shape[-1] == N else restrict(x, N)
        return (rs(turb[di][None])[0], rs(turb[pi][None])[0],
                rs(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]])))
    BG = {N: bg(N) for N in (32, 64)}
    TGT = {N: (lambda t: t / jnp.sum(t) * jnp.sum(BG[N][0]))(load_image("logo.png", N))
           for N in (32, 64)}

    def make_vg(N):
        rho, p, B = BG[N]; r = rvN[N]; tg = TGT[N]
        def loss(theta, mask):
            v = lowpass(theta, mask)
            s = build(cfgB[N], r, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
            proj = jnp.sum(time_integration(s, cfgB[N], inv, r)[di], axis=2)
            return jnp.mean((proj - tg) ** 2)
        return jax.jit(jax.value_and_grad(loss))
    VG = {N: make_vg(N) for N in (32, 64)}

    rho64, p64, B64 = BG[64]
    def loss64_fwd(theta, mask):                       # diagnostic 64³ forward (no grad)
        v = lowpass(theta, mask)
        s = build(cfg64f, rv, rho64, (v[0], v[1], v[2]), p64, (B64[0], B64[1], B64[2]))
        proj = jnp.sum(time_integration(s, cfg64f, inv, rv)[di], axis=2)
        return jnp.mean((proj - TGT[64]) ** 2)
    FL64 = jax.jit(loss64_fwd)

    theta0 = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])    # turbulent 'now' warm start
    FULL = 64                                                   # kcut covering all 64³ modes

    # Warm up (compile) every jitted fn ONCE, untimed -- compile is a one-off (cached on
    # disk), so folding it into a timed step would unfairly penalise whichever method
    # compiles first.  After this, all timed steps are steady-state.
    tw = time.time()
    jax.block_until_ready(VG[64](theta0, kmask(64, FULL)))
    jax.block_until_ready(VG[32](restrict(theta0, 32), kmask(32, args.coarse_cut)))
    jax.block_until_ready(FL64(theta0, kmask(64, args.coarse_cut)))
    print(f"warmup/compile done ({time.time()-tw:.0f}s)", flush=True)

    def stages_for(method):
        if method == "naive":
            return [(FULL, TOTAL_STEPS, 64)]
        grid = (lambda kc: 32 if (method == "multigrid" and kc <= args.coarse_cut) else 64)
        return [(kc, st, grid(kc)) for (kc, st) in SCHED]

    def run(method):
        stages = stages_for(method)
        N0 = stages[0][2]
        theta = restrict(theta0, N0) if N0 != 64 else theta0
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)
        cum = 0.0; tstep = 0; rec = []                 # (cumtime, loss64)
        for (kc, steps, N) in stages:
            if theta.shape[-1] != N:                   # prolongation-free transfer
                theta = prolong(theta, N); m = prolong(m, N); vv = prolong(vv, N)
            mask = kmask(N, kc)
            for s in range(steps):
                t0 = time.time()
                l, g = VG[N](theta, mask)
                tstep += 1
                m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
                theta = theta - lr * (m / (1 - b1 ** tstep)) / (jnp.sqrt(vv / (1 - b2 ** tstep)) + eps)
                cum += time.time() - t0                # optimisation time only
                if N == 64:
                    loss64 = float(l)
                elif s % args.diag_every == 0 or s == steps - 1:
                    loss64 = float(FL64(prolong(theta, 64), kmask(64, kc)))   # diagnostic
                else:
                    loss64 = None
                if loss64 is not None:
                    rec.append((cum, loss64))
                    print(f"   [{method:9s}] kc={kc:2d}(N={N}) step {tstep:3d}: "
                          f"loss64={loss64:.4e} cum={cum:.0f}s", flush=True)
        return np.array(rec)

    methods = ("naive", "k-windowing", "multigrid") if args.only == "all" else (args.only,)
    out = {}
    for method in methods:
        print(f"--- {method} ---", flush=True)
        out[method.replace("-", "_")] = run(method)
    np.savez(args.out, sched=np.array(SCHED), coarse_cut=args.coarse_cut, **out)
    print(f"\nwrote {args.out}", flush=True)
    for k, v in out.items():
        print(f"  {k:12s} final loss64={v[-1,1]:.4e} @ {v[-1,0]:.0f}s "
              f"(reached {v[-1,1]:.2e} in {len(v)} logged pts)", flush=True)


if __name__ == "__main__":
    main()
