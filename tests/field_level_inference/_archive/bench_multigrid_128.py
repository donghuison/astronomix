#!/usr/bin/env python
"""Twin benchmark v2: FULL 3D density field as the observable (not a projection),
multigrid 64³-coarse -> 128³-fine vs direct-128³, SAME k-expansion schedule.

Motivation (from the projection twin `bench_multigrid_vs_direct.py`): a single
z-projection of density is IC-DEGENERATE -- projection MSE -> 0 while
||v-v_true|| stays ~1.  Two changes here:
  (1) observable = the full 3D density field rho(x, t_end) -- maximally informative,
      a clean identifiability control: is the IC recoverable in principle?
  (2) multigrid ratio 64³->128³ (8x cells) instead of 32³->64³ (~3x), so the
      cheap coarse stage can actually amortise.

Twin: known band-limited v_true (|k|<k_true) on a turbulent background gives the
target full field at t_end (and its 3D spectral restriction for the coarse stage).
Both methods recover v from zero; they differ ONLY in the per-stage grid, set by
the schedule's grid tag.  IC error is measured at the fine grid (band-limited v
prolonged).  cumtime counts only optimisation steps.  astx/GPU.

Schedule format: comma-separated  kcut:steps:grid  (grid in cells).
  multigrid: uses the tagged grid per stage (coarse bands on 64³, polish on 128³);
  direct:    forces every stage onto --fine-N.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--coarse-N", type=int, default=64)
ap.add_argument("--fine-N", type=int, default=128)
ap.add_argument("--k-true", type=int, default=24, help="v_true band limit (|k|<k_true).")
ap.add_argument("--sched", type=str, default="12:6:64,24:8:64,24:8:128",
                help="kcut:steps:grid per stage (multigrid uses grid; direct forces fine-N).")
ap.add_argument("--num-checkpoints", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--probe", action="store_true",
                help="time one fwd+grad on each grid then exit (no optimisation).")
ap.add_argument("--budget", type=float, default=0.0,
                help="per-method wall-time budget [s]; stop a method once exceeded (0=off).")
ap.add_argument("--skip-direct", action="store_true",
                help="run only the multigrid method (cheap 64³ degeneracy check).")
ap.add_argument("--out", type=str, default="bench_mg_128.npz")
ap.add_argument("--t-mult", type=float, default=1.0,
                help="t_end = t_mult * (0.5/rms); >1 = longer horizon (identifiability sweep).")
ap.add_argument("--observable", choices=["density", "fullstate"], default="density",
                help="density: z-... no, full 3D rho.  fullstate: all state channels "
                     "(rho,v,B,p) per-channel-normalised -- control isolating density-only.")
args = ap.parse_args()

SCHED = [(int(a.split(":")[0]), int(a.split(":")[1]), int(a.split(":")[2]))
         for a in args.sched.split(",")]
GRIDS = sorted({g for (_, _, g) in SCHED} | {args.fine_N})

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


def helmholtz(v):
    """Split a (3,N,N,N) real velocity into (compressive, solenoidal) parts via
    the Fourier Leray projector.  Density observables constrain mainly the
    compressive (longitudinal) part; the solenoidal part is the inverse's null
    space, so tracking the two IC errors separately is the decisive diagnostic."""
    N = v.shape[-1]
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = [jnp.asarray(a) for a in np.meshgrid(k, k, k, indexing="ij")]
    k2 = KX**2 + KY**2 + KZ**2
    k2 = jnp.where(k2 == 0, 1.0, k2)
    vh = jnp.fft.fftn(v, axes=(-3, -2, -1))
    kdotv = KX * vh[0] + KY * vh[1] + KZ * vh[2]
    comp_h = jnp.stack([KX, KY, KZ]) * (kdotv / k2)        # longitudinal
    v_comp = jnp.fft.ifftn(comp_h, axes=(-3, -2, -1)).real
    return v_comp, v - v_comp


def fields(turb, rv, N):
    """Background rho,p,v,B sampled at resolution N (restrict/prolong from turb's grid)."""
    di, pi, vi, mi = rv.density_index, rv.pressure_index, rv.velocity_index, rv.magnetic_index
    rs = lambda x: x if x.shape[-1] == N else (restrict(x, N) if N < x.shape[-1] else prolong(x, N))
    return (rs(turb[di][None])[0], rs(turb[pi][None])[0],
            rs(jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])),
            rs(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]])))


def main():
    Nf = args.fine_N
    print(f"=== twin-128 bench  FULL-FIELD  k_true={args.k_true}  sched={SCHED}  "
          f"coarse={args.coarse_N} fine={Nf} ===", flush=True)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    # turbulent background generated directly at the fine grid (genuine high-k structure)
    cfgg = make_cfg(Nf, True, False); rv = get_registered_variables(cfgg)
    sh = (Nf,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    vi, di = rv.velocity_index, rv.density_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = args.t_mult * 0.5 / max(rms, 1e-12)        # t_mult * (½ crossing time)
    inv = params._replace(t_end=t_end)

    # --- twin truth: band-limited random v_true (|k|<k_true) at fine grid, amp ~ rms ---
    key = jax.random.key(7)
    v_true = lowpass(jax.random.normal(key, (3,) + sh), kmask(Nf, args.k_true))
    v_true = v_true * (rms / jnp.sqrt(jnp.mean(v_true**2)))
    cfg_ff = finalize_config(make_cfg(Nf, False, False), turb.shape)
    rhoF, pF, _, BF = fields(turb, rv, Nf)
    s_true = build(cfg_ff, rv, rhoF, (v_true[0], v_true[1], v_true[2]), pF, (BF[0], BF[1], BF[2]))
    state_fine = time_integration(s_true, cfg_ff, inv, rv)            # full evolved state
    target_fine = state_fine if args.observable == "fullstate" else state_fine[di]
    vt_norm = float(jnp.sqrt(jnp.sum(v_true**2)))
    print(f"turb+twin done  rms={rms:.4f}  t_end={t_end:.4f}  |v_true|={vt_norm:.3f}  "
          f"obs={args.observable}  rho[{float(state_fine[di].min()):.4f},"
          f"{float(state_fine[di].max()):.4f}]", flush=True)

    # per-grid backgrounds, backward configs, and restricted targets (density or full state)
    cfgB = {N: finalize_config(make_cfg(N, False, True), (11,) + (N,) * 3) for N in GRIDS}
    rvN = {N: get_registered_variables(make_cfg(N, False, True)) for N in GRIDS}
    bg = {N: fields(turb, rv, N) for N in GRIDS}
    def rest_tgt(T, N):
        if N == Nf: return T
        return restrict(T, N) if T.ndim == 4 else restrict(T[None], N)[0]
    tgt = {N: rest_tgt(target_fine, N) for N in GRIDS}

    def make_loss(N):
        rho, p, _, B = bg[N]; r = rvN[N]; T = tgt[N]
        # O(1) relative MSE -- raw rho-field MSE ~1e-6 (rho~0.013); per-elem grad below
        # Adam eps starves the optimiser, so normalise by target power.
        if args.observable == "density":
            tnorm = jnp.mean(T ** 2)
            def loss(theta, mask):
                v = lowpass(theta, mask)
                s = build(cfgB[N], r, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
                fld = time_integration(s, cfgB[N], inv, r)[di]
                return jnp.mean((fld - T) ** 2) / tnorm
        else:  # full-state control: per-channel-normalised MSE over all vars (incl. velocity)
            tnorm = jnp.mean(T ** 2, axis=(-3, -2, -1)) + 1e-30
            def loss(theta, mask):
                v = lowpass(theta, mask)
                s = build(cfgB[N], r, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
                full = time_integration(s, cfgB[N], inv, r)
                return jnp.mean(jnp.mean((full - T) ** 2, axis=(-3, -2, -1)) / tnorm)
        return jax.jit(jax.value_and_grad(loss))
    VG = {N: make_loss(N) for N in GRIDS}

    vt_c, vt_s = helmholtz(v_true)
    vtc_norm = float(jnp.sqrt(jnp.sum(vt_c ** 2)))
    vts_norm = float(jnp.sqrt(jnp.sum(vt_s ** 2)))
    print(f"v_true Helmholtz: compressive {vtc_norm/vt_norm:.3f}  "
          f"solenoidal {vts_norm/vt_norm:.3f} (of |v_true|)", flush=True)

    def ic_err(theta, mask, N):
        v = lowpass(theta, mask)
        v64 = v if N == Nf else prolong(v, Nf)
        tot = float(jnp.sqrt(jnp.sum((v64 - v_true) ** 2)) / vt_norm)
        vc, vs = helmholtz(v64)
        ec = float(jnp.sqrt(jnp.sum((vc - vt_c) ** 2)) / max(vtc_norm, 1e-30))
        es = float(jnp.sqrt(jnp.sum((vs - vt_s) ** 2)) / max(vts_norm, 1e-30))
        corr = float(jnp.sum(v64 * v_true) / (jnp.sqrt(jnp.sum(v64**2)) * vt_norm + 1e-30))
        return tot, ec, es, corr

    if args.probe:
        for N in GRIDS:
            th = jnp.zeros((3,) + (N,) * 3); mk = kmask(N, args.k_true)
            l, g = jax.block_until_ready(VG[N](th, mk))            # compile
            t0 = time.time(); l, g = jax.block_until_ready(VG[N](th, mk))
            print(f"   PROBE N={N}: step time {time.time()-t0:.1f}s  L={float(l):.3e}", flush=True)
        return

    def run(method):
        def grid_for(kcut, grid):
            return args.fine_N if method == "direct" else grid
        N0 = grid_for(*SCHED[0][::2])
        theta = jnp.zeros((3,) + (N0,) * 3)
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)
        cum = 0.0; tstep = 0
        rec = []  # (cumtime, term_err, ic_tot, ic_comp, ic_sol, corr)
        for (kcut, steps, grid) in SCHED:
            N = grid_for(kcut, grid)
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
                tot, ec, es, corr = ic_err(theta, mask, N)
                rec.append((cum, float(l), tot, ec, es, corr))
                print(f"   [{method}] kcut={kcut:2d}(N={N}) step {tstep:3d}: "
                      f"L={float(l):.3e} IC={tot:.3f} (comp={ec:.3f} sol={es:.3f} "
                      f"corr={corr:+.3f}) cum={cum:.0f}s", flush=True)
                if args.budget and cum >= args.budget:
                    print(f"   [{method}] budget {args.budget:.0f}s reached -> stop", flush=True)
                    return np.array(rec)
        return np.array(rec)

    print("--- MULTIGRID (coarse on 64³) ---", flush=True)
    rec_mg = run("multigrid")
    rec_dir = np.empty((0, 3))
    if not args.skip_direct:
        print("--- DIRECT (all 128³) ---", flush=True)
        rec_dir = run("direct")
    np.savez(args.out, mg=rec_mg, direct=rec_dir,
             sched=np.array([(k, s) for (k, s, g) in SCHED]),
             grids=np.array([g for (_, _, g) in SCHED]),
             k_true=args.k_true, coarse_N=args.coarse_N, fine_N=Nf)
    msg = f"\nwrote {args.out}  (mg final: L={rec_mg[-1,1]:.3e} IC={rec_mg[-1,2]:.3f} @ {rec_mg[-1,0]:.0f}s"
    if rec_dir.size:
        msg += f"; direct final: L={rec_dir[-1,1]:.3e} IC={rec_dir[-1,2]:.3f} @ {rec_dir[-1,0]:.0f}s"
    print(msg + ")", flush=True)


if __name__ == "__main__":
    main()
