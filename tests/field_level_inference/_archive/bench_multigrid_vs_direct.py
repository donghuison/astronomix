#!/usr/bin/env python
"""Twin benchmark: prolongation-free multigrid warm-start vs direct 64³, SAME
k-expansion schedule.  Plots terminal error (loss) + IC error (||v - v_true||)
vs cumulative optimisation wall-time.

Twin: a known band-limited v_true (|k|<k_true) on the turbulent background gives
the target z-projection at t_end (and its 2D coarse restriction for the 32³
stage).  Both methods recover v from zero; they differ ONLY in the grid used for
the coarse bands:
  multigrid: bands with k_cut <= k_Nyq_lo - gap on 32³ (cheap, ~17x), rest on 64³;
  direct:    the identical schedule entirely on 64³.
IC error is measured at 64³ (band-limited v prolonged); cumtime counts only the
optimisation steps (diagnostic forwards excluded).  astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--k-true", type=int, default=16, help="v_true band limit (|k|<k_true).")
ap.add_argument("--sched", type=str, default="8:15,16:10,32:15",
                help="k_cut:steps schedule (grid chosen per method).")
ap.add_argument("--gap", type=int, default=8, help="multigrid: k_cut<=16-gap runs on 32³.")
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
args = ap.parse_args()

SCHED = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in args.sched.split(",")]

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
KNYQ_LO = 16  # 32³ Nyquist


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


def kmask(N, kcut):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    return jnp.asarray((np.sqrt(KX**2 + KY**2 + KZ**2) < kcut).astype(np.float64))


def lowpass(theta, mask):
    return jnp.fft.ifftn(jnp.fft.fftn(theta, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real


def restrict2d(img, n_lo):
    n_hi = img.shape[-1]
    F = jnp.fft.fftshift(jnp.fft.fftn(img))
    c0, h = n_hi // 2, n_lo // 2
    F = F[c0 - h:c0 + h, c0 - h:c0 + h]
    F = F.at[0, :].set(0).at[:, 0].set(0)
    return jnp.fft.ifftn(jnp.fft.ifftshift(F)).real * (n_lo / n_hi) ** 2


def fields(turb, rv, N):
    di, pi, vi, mi = rv.density_index, rv.pressure_index, rv.velocity_index, rv.magnetic_index
    rs = lambda x: x if x.shape[-1] == N else (restrict(x, N) if N < x.shape[-1] else prolong(x, N))
    return (rs(turb[di][None])[0], rs(turb[pi][None])[0],
            rs(jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])),
            rs(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]])))


def main():
    print(f"=== twin bench  k_true={args.k_true}  sched={SCHED}  gap={args.gap} ===", flush=True)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    cfgg = make_cfg(64, True, False); rv = get_registered_variables(cfgg)
    sh = (64,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    vi, di = rv.velocity_index, rv.density_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    inv = params._replace(t_end=t_end)

    # --- twin truth: band-limited random v_true (|k|<k_true), amp ~ turbulent rms ---
    key = jax.random.key(7)
    raw = jax.random.normal(key, (3,) + sh)
    v_true = lowpass(raw, kmask(64, args.k_true))
    v_true = v_true * (rms / jnp.sqrt(jnp.mean(v_true**2)))
    cfg_f64 = finalize_config(make_cfg(64, False, False), turb.shape)
    rho64, p64, _, B64 = fields(turb, rv, 64)
    s_true = build(cfg_f64, rv, rho64, (v_true[0], v_true[1], v_true[2]), p64, (B64[0], B64[1], B64[2]))
    target64 = jnp.sum(time_integration(s_true, cfg_f64, inv, rv)[di], axis=2)
    target32 = restrict2d(target64, 32)
    vt_norm = float(jnp.sqrt(jnp.sum(v_true**2)))
    print(f"turb+twin done  rms={rms:.4f}  t_end={t_end:.4f}  |v_true|={vt_norm:.3f}", flush=True)

    # per-grid backgrounds + backward configs
    cfgB = {N: finalize_config(make_cfg(N, False, True), (11,) + (N,) * 3) for N in (32, 64)}
    rvN = {N: get_registered_variables(make_cfg(N, False, True)) for N in (32, 64)}
    bg = {N: fields(turb, rv, N) for N in (32, 64)}
    tgt = {32: target32, 64: target64}

    def make_loss(N):
        rho, p, _, B = bg[N]; r = rvN[N]
        def loss(theta, mask):
            v = lowpass(theta, mask)
            s = build(cfgB[N], r, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
            proj = jnp.sum(time_integration(s, cfgB[N], inv, r)[di], axis=2)
            return jnp.mean((proj - tgt[N]) ** 2)
        return jax.jit(jax.value_and_grad(loss))
    VG = {N: make_loss(N) for N in (32, 64)}

    def ic_err(theta, mask, N):
        v = lowpass(theta, mask)
        v64 = v if N == 64 else prolong(v, 64)
        return float(jnp.sqrt(jnp.sum((v64 - v_true) ** 2)) / vt_norm)

    def run(method):
        # grid per band
        def grid_for(kcut):
            if method == "multigrid" and kcut <= KNYQ_LO - args.gap:
                return 32
            return 64
        theta = jnp.zeros((3, 32, 32, 32)) if grid_for(SCHED[0][0]) == 32 else jnp.zeros((3, 64, 64, 64))
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)
        cum = 0.0; tstep = 0
        rec = []  # (cumtime, term_err64, ic_err)
        for (kcut, steps) in SCHED:
            N = grid_for(kcut)
            if theta.shape[-1] != N:                  # exact prolongation-free transfer
                theta = prolong(theta, N); m = prolong(m, N); vv = prolong(vv, N)
            mask = kmask(N, kcut)
            for s in range(steps):
                t0 = time.time()
                l, g = VG[N](theta, mask)
                tstep += 1
                m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
                theta = theta - lr * (m / (1 - b1 ** tstep)) / (jnp.sqrt(vv / (1 - b2 ** tstep)) + eps)
                cum += time.time() - t0            # only optimisation time
                ie = ic_err(theta, mask, N)
                rec.append((cum, float(l), ie))
                if s == steps - 1:
                    print(f"   [{method}] kcut={kcut:2d}(N={N}) step {tstep:3d}: "
                          f"L={float(l):.3e} IC={ie:.3f} cum={cum:.0f}s", flush=True)
        return np.array(rec)

    print("--- MULTIGRID (coarse on 32³) ---", flush=True)
    rec_mg = run("multigrid")
    print("--- DIRECT (all 64³) ---", flush=True)
    rec_dir = run("direct")
    np.savez("bench_mg_vs_direct.npz", mg=rec_mg, direct=rec_dir,
             sched=np.array(SCHED), k_true=args.k_true, gap=args.gap)
    print(f"\nwrote bench_mg_vs_direct.npz  "
          f"(mg final: L={rec_mg[-1,1]:.3e} IC={rec_mg[-1,2]:.3f} @ {rec_mg[-1,0]:.0f}s; "
          f"direct final: L={rec_dir[-1,1]:.3e} IC={rec_dir[-1,2]:.3f} @ {rec_dir[-1,0]:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
