# =============================================================================
#  Kelvin–Helmholtz Instability: Critical Reynolds Number Demonstration
#  Based on Roediger et al. (2013), MNRAS (arXiv:1309.2635)
#
#  Two simulations of a dense slab (stream) flowing through a dilute
#  background, one above and one below the critical Reynolds number:
#
#     Re_crit = 880 / Δ       (Eq. 22, for constant kinematic viscosity)
#     Δ = (ρ_cold + ρ_hot)² / (ρ_cold ρ_hot)
#
#  with the viscous growth time (Eq. 21):
#     τ_KH,visc = τ_KH,inv × [1 + Re₀ / (Re − Re_crit)]
#
#  Case A  (Re >> Re_crit):  viscosity negligible  → rolls develop normally
#  Case B  (Re <  Re_crit):  viscosity dominates   → KHI suppressed
#
#  NOTE: The code uses constant *dynamic* viscosity μ.  For density
#  contrast δ > 1 this means the kinematic viscosity ν = μ/ρ differs
#  between the two layers (lower in the dense stream, higher in the
#  dilute background), analogous in spirit to the Spitzer-viscosity
#  regime discussed in §4.3 of the paper.
# =============================================================================

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os
os.makedirs("figures", exist_ok=True)

# ── numerics ─────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from timeit import default_timer as timer

# ── plotting ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation

# ── astronomix ───────────────────────────────────────────────────────────────
from astronomix import (
    SimulationConfig,
    get_helper_data,
    SimulationParams,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, finalize_config
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)


# =============================================================================
#  §1  PHYSICAL PARAMETERS
# =============================================================================
#
#  Same slab-in-a-box setup as the companion Mach-number demo: a dense
#  stream (subscript s) in pressure equilibrium with a hot dilute
#  background (subscript b).  Here the stream velocity is *fixed* at a
#  subsonic Mach number (Mb = 0.5 in the background), and we vary the
#  viscosity to move above / below the critical Reynolds number.
#
#  Re = λ U ρ_b / μ     (background-layer Reynolds number)
#
#  For δ = 10 with constant kinematic ν:
#     Δ ≈ 12.1,   Re_crit ≈ 73   (Eq. 22)
#
#  For δ = 10 with constant dynamic μ (our case), the background
#  (hot, low-density) layer controls suppression. From the paper's
#  Spitzer-viscosity results (§4.3):
#     Re_crit,hot ≈ 10–30  for Dρ = 10
#
#  We bracket both regimes conservatively.
# ─────────────────────────────────────────────────────────────────────────

gamma = 5.0 / 3.0        # adiabatic index
delta = 10.0              # density contrast  ρ_s / ρ_b

rho_b = 1.0               # background density
rho_s = delta * rho_b     # stream density  (= 10)
P0    = 1.0               # uniform pressure  (pressure equilibrium)

c_b = float(jnp.sqrt(gamma * P0 / rho_b))   # background sound speed ≈ 1.291
c_s = float(jnp.sqrt(gamma * P0 / rho_s))   # stream   sound speed ≈ 0.408

# ── shear velocity: Mach 0.5 in the background (same for both cases) ─────
Mb = 0.5
V_stream = Mb * c_b     # stream velocity (background at rest)
U_shear  = V_stream      # total velocity difference across interface

# ── perturbation ─────────────────────────────────────────────────────────
box_size = 1.0
n_waves  = 5              # wavelengths in the box
lam      = box_size / n_waves   # perturbation wavelength λ = 0.2

# ── inviscid KHI growth time (Roediger Eq. 2) ───────────────────────────
Drho  = rho_s / rho_b
Delta = (Drho + 1.0) ** 2 / Drho                   # Δ ≈ 12.1
tau_KH_inv = float(jnp.sqrt(Delta) / (2.0 * jnp.pi) * lam / U_shear)

# ── critical Reynolds number (Roediger Eqs. 22–23, constant ν) ──────────
Re_crit_const_nu = 880.0 / Delta                    # ≈ 73
Re0_const_nu     = 1320.0 / jnp.sqrt(Delta)         # ≈ 379

# ── two cases straddling the critical region ─────────────────────────────
Re_A = 3000     # high Re  → viscosity negligible  → KHI develops
Re_B = 20       # low  Re  → viscosity dominates   → KHI suppressed

# dynamic viscosity μ  (constant throughout the domain)
#   Re_b = λ U ρ_b / μ   →   μ = λ U ρ_b / Re_b
mu_A = lam * U_shear * rho_b / Re_A
mu_B = lam * U_shear * rho_b / Re_B

# kinematic viscosity in each layer
nu_b_A, nu_s_A = mu_A / rho_b, mu_A / rho_s
nu_b_B, nu_s_B = mu_B / rho_b, mu_B / rho_s

# predicted viscous growth time for Case A (Eq. 21 with const-ν formulae)
tau_visc_A = tau_KH_inv * (1.0 + float(Re0_const_nu) / (Re_A - Re_crit_const_nu))

print("=" * 65)
print(" Viscous KHI — Roediger et al. (2013)")
print("=" * 65)
print(f"  γ = {gamma:.4f},   δ = {delta:.0f}")
print(f"  c_b = {c_b:.4f},   c_s = {c_s:.4f}")
print(f"  Mb = {Mb:.2f},   U = {U_shear:.4f}")
print(f"  λ = {lam:.4f},   Δ = {Delta:.2f}")
print(f"  τ_KH,inv = {tau_KH_inv:.4f}")
print(f"  Re_crit (const ν, Eq. 22) ≈ {Re_crit_const_nu:.0f}")
print(f"  Re₀     (const ν, Eq. 23) ≈ {float(Re0_const_nu):.0f}")
print()
print(f"  Case A   Re_b = {Re_A:>5d}   >>  Re_crit")
print(f"           μ = {mu_A:.2e},  ν_b = {nu_b_A:.2e},  ν_s = {nu_s_A:.2e}")
print(f"           predicted τ_visc/τ_inv ≈ {tau_visc_A/tau_KH_inv:.2f}")
print()
print(f"  Case B   Re_b = {Re_B:>5d}   <<  Re_crit")
print(f"           μ = {mu_B:.2e},  ν_b = {nu_b_B:.2e},  ν_s = {nu_s_B:.2e}")
print(f"           KHI expected to be SUPPRESSED")
print("=" * 65)


# =============================================================================
#  §2  SIMULATION CONFIGURATION
# =============================================================================
num_cells     = 512
num_timesteps = 30000     # upper bound; CFL-adaptive stepping stops at t_end
t_end         = 3.0       # ≈ 35 × τ_KH,inv — long enough to see suppression

config = SimulationConfig(
    solver_mode       = FINITE_DIFFERENCE,
    progress_bar      = True,
    dimensionality    = 2,
    box_size          = box_size,
    num_cells         = num_cells,
    num_timesteps     = num_timesteps,
    diffusion         = True,          # ← enable viscous terms
    boundary_settings = BoundarySettings(
        x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    limiter           = DOUBLE_MINMOD,
    return_snapshots  = True,
    num_snapshots     = 100,
)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)


# =============================================================================
#  §3  GRID, SLAB GEOMETRY, AND PERTURBATION
# =============================================================================
#
#  Slab (= stream) centred at y = 0.5, half-width Rs = 0.1
#  Smoothed interface with tanh on scale σ  (cf. Roediger Eq. 14).
#  Perturbation: transverse velocity  v_y = A · sin(k·x)
# ─────────────────────────────────────────────────────────────────────────

dx = box_size / num_cells
x  = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, num_cells)
y  = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, num_cells)
X, Y = jnp.meshgrid(x, y, indexing="ij")

y_center = 0.5
Rs       = 0.1           # slab half-width
sigma    = 0.008          # smoothing scale  (σ/Δx ≈ 4 at 512)

# perturbation
k_pert = 2.0 * jnp.pi * n_waves / box_size
A_pert = 0.01 * c_s      # ~1 % of stream sound speed

print(f"\n  Slab:  y ∈ [{y_center-Rs:.2f}, {y_center+Rs:.2f}],  Rs = {Rs}")
print(f"  Smoothing:  σ = {sigma},  σ/Δx = {sigma/dx:.1f}")
print(f"  Perturbation:  {n_waves} waves,  λ = {lam:.3f},  K = k·Rs = {float(k_pert*Rs):.4f}")
print(f"  τ_KH,inv = {tau_KH_inv:.4f}")
print(f"  Simulation end:  t_end = {t_end}\n")

# ── smoothed slab profile ────────────────────────────────────────────────
def slab_profile(f_b, f_s):
    """Tanh transition from f_b (background) to f_s (stream)."""
    return f_b + 0.5 * (f_s - f_b) * (
        1.0 + jnp.tanh((Rs - jnp.abs(Y - y_center)) / sigma)
    )


# ── build initial primitive state (same for both Re cases) ───────────────
rho = slab_profile(rho_b, rho_s)
vx  = slab_profile(0.0,   V_stream)
vy  = A_pert * jnp.sin(k_pert * X)
p   = P0 * jnp.ones_like(X)

initial_state = construct_primitive_state(
    config               = config,
    registered_variables = registered_variables,
    density              = rho,
    velocity_x           = vx,
    velocity_y           = vy,
    gas_pressure         = p,
)


# =============================================================================
#  §4  FINALIZE & SET UP PARAMS FOR EACH CASE
# =============================================================================
config = finalize_config(config, initial_state.shape)

params_A = SimulationParams(
    t_end            = t_end,
    C_cfl            = 0.4,
    minimum_density  = 1e-8,
    minimum_pressure = 1e-10,
    viscosity        = mu_A,
)
params_B = SimulationParams(
    t_end            = t_end,
    C_cfl            = 0.4,
    minimum_density  = 1e-8,
    minimum_pressure = 1e-10,
    viscosity        = mu_B,
)


# =============================================================================
#  §5  RUN SIMULATIONS
# =============================================================================
print("━" * 65)
print(f"  Running Case A   Re_b = {Re_A}  (low viscosity) …")
print("━" * 65)
t0 = timer()
result_A = time_integration(initial_state, config, params_A, registered_variables)
print(f"  ✓ Case A finished in {timer()-t0:.1f} s\n")

print("━" * 65)
print(f"  Running Case B   Re_b = {Re_B}  (high viscosity) …")
print("━" * 65)
t0 = timer()
result_B = time_integration(initial_state, config, params_B, registered_variables)
print(f"  ✓ Case B finished in {timer()-t0:.1f} s\n")


# =============================================================================
#  §6  GROWTH-RATE DIAGNOSTIC  (max |v_y|)
# =============================================================================
#
#  Roediger et al. track v_{y,max} over time (Figs. 3, 7, 10).
#  Expect:
#    Case A — exponential growth ∝ exp(t / τ_KH), then saturation
#    Case B — no growth; initial perturbation decays due to dissipation
# ─────────────────────────────────────────────────────────────────────────
n_snaps    = min(len(result_A.states), len(result_B.states))
snap_times = jnp.linspace(0, t_end, n_snaps)

vy_idx = registered_variables.velocity_index.y
p_idx  = registered_variables.pressure_index

# slab region in y  (extend slightly beyond slab for safety)
j_slab_lo = int((y_center - 1.5 * Rs) * num_cells)
j_slab_hi = int((y_center + 1.5 * Rs) * num_cells)

vymax_A, vymax_B = [], []
amp_A, amp_B     = [], []     # pressure perturbation amplitude
for i in range(n_snaps):
    # max |v_y| in the whole domain (Roediger's diagnostic)
    vymax_A.append(float(jnp.max(jnp.abs(result_A.states[i][vy_idx]))))
    vymax_B.append(float(jnp.max(jnp.abs(result_B.states[i][vy_idx]))))

    # mean |P − P₀| in the slab region (complementary diagnostic)
    dP_A = jnp.abs(result_A.states[i][p_idx, :, j_slab_lo:j_slab_hi] - P0)
    dP_B = jnp.abs(result_B.states[i][p_idx, :, j_slab_lo:j_slab_hi] - P0)
    amp_A.append(float(jnp.mean(dP_A)))
    amp_B.append(float(jnp.mean(dP_B)))

vymax_A = jnp.array(vymax_A)
vymax_B = jnp.array(vymax_B)
amp_A   = jnp.array(amp_A)
amp_B   = jnp.array(amp_B)

# ── plot: v_{y,max} vs time (cf. Roediger Fig. 3, bottom panel) ──────────
fig_gr, (ax_vy, ax_dp) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

# normalise time by inviscid growth time
t_norm = snap_times / tau_KH_inv

# v_y,max panel
ax_vy.semilogy(t_norm, vymax_A, "C0-",  lw=2,
               label=f"Case A: Re$_b={Re_A}$  (low $\\nu$)")
ax_vy.semilogy(t_norm, vymax_B, "C3--", lw=2,
               label=f"Case B: Re$_b={Re_B}$  (high $\\nu$)")
ax_vy.set_ylabel(r"$|v_y|_{\rm max}$", fontsize=12)
ax_vy.set_title(
    f"Viscous KHI growth diagnostic  —  "
    f"$\\delta={delta:.0f}$,  Mach$={Mb}$,  "
    f"Re$_{{\\rm crit}}\\approx{Re_crit_const_nu:.0f}$  (const $\\nu$)",
    fontsize=13,
)
ax_vy.legend(fontsize=11)
ax_vy.grid(True, which="both", alpha=0.3)

# pressure perturbation panel
ax_dp.semilogy(t_norm, amp_A, "C0-",  lw=2,
               label=f"Case A: Re$_b={Re_A}$")
ax_dp.semilogy(t_norm, amp_B, "C3--", lw=2,
               label=f"Case B: Re$_b={Re_B}$")
ax_dp.set_xlabel(r"$t\;/\;\tau_{\rm KH,inv}$", fontsize=12)
ax_dp.set_ylabel(r"$\langle\,|P - P_0|\,\rangle$  (slab region)", fontsize=12)
ax_dp.legend(fontsize=11)
ax_dp.grid(True, which="both", alpha=0.3)

fig_gr.tight_layout()
fig_gr.savefig("figures/khi_viscous_growth_rate.png", dpi=150)
print("Saved → figures/khi_viscous_growth_rate.png")
plt.close(fig_gr)


# =============================================================================
#  §7  FINAL-STATE COMPARISON  (density, v_y, pressure perturbation)
# =============================================================================
fig_fs, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)

y_lo, y_hi = 0.15, 0.85
extent = [0, box_size, y_lo, y_hi]
jlo = int(y_lo * num_cells)
jhi = int(y_hi * num_cells)

def sl(state):
    return state[:, :, jlo:jhi]

sA = result_A.states[-1]
sB = result_B.states[-1]

# ── density ──
rho_norm = LogNorm(vmin=0.5, vmax=15, clip=True)
axes[0, 0].imshow(sl(sA)[0].T, norm=rho_norm, cmap="viridis",
                  origin="lower", extent=extent)
axes[1, 0].imshow(sl(sB)[0].T, norm=rho_norm, cmap="viridis",
                  origin="lower", extent=extent)
axes[0, 0].set_title("Density  $\\rho$", fontsize=12)

# ── transverse velocity ──
vy_lim = float(jnp.max(jnp.abs(sl(sA)[vy_idx])))
vy_lim = max(vy_lim, float(jnp.max(jnp.abs(sl(sB)[vy_idx]))))
axes[0, 1].imshow(sl(sA)[vy_idx].T, cmap="RdBu_r", origin="lower",
                  extent=extent, vmin=-vy_lim, vmax=vy_lim)
axes[1, 1].imshow(sl(sB)[vy_idx].T, cmap="RdBu_r", origin="lower",
                  extent=extent, vmin=-vy_lim, vmax=vy_lim)
axes[0, 1].set_title("Transverse velocity  $v_y$", fontsize=12)

# ── pressure perturbation ──
dP_A = sl(sA)[p_idx] - P0
dP_B = sl(sB)[p_idx] - P0
dp_lim = float(jnp.max(jnp.abs(dP_A)))
dp_lim = max(dp_lim, float(jnp.max(jnp.abs(dP_B))), 1e-6)
axes[0, 2].imshow(dP_A.T, cmap="RdBu_r", origin="lower",
                  extent=extent, vmin=-dp_lim, vmax=dp_lim)
axes[1, 2].imshow(dP_B.T, cmap="RdBu_r", origin="lower",
                  extent=extent, vmin=-dp_lim, vmax=dp_lim)
axes[0, 2].set_title("Pressure perturbation  $P - P_0$", fontsize=12)

for ax in axes[0, :]:
    ax.set_aspect("equal")
for ax in axes[1, :]:
    ax.set_aspect("equal")
    ax.set_xlabel("$x$  (flow)", fontsize=11)

axes[0, 0].set_ylabel(
    f"Case A   Re$_b={Re_A}$\n$y$", fontsize=11)
axes[1, 0].set_ylabel(
    f"Case B   Re$_b={Re_B}$\n$y$", fontsize=11)

# slab boundaries
for row in axes:
    for ax in row:
        ax.axhline(y_center - Rs, color="w", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(y_center + Rs, color="w", ls="--", lw=0.6, alpha=0.5)

fig_fs.suptitle(
    f"Final state at $t = {t_end}$   —   $\\delta={delta:.0f}$,  "
    f"Mach$={Mb}$,  $\\gamma={gamma:.2f}$\n"
    f"Re$_{{\\rm crit}}\\approx{Re_crit_const_nu:.0f}$  "
    f"(Roediger+2013 Eq. 22, const $\\nu$)",
    fontsize=13, y=1.00,
)
fig_fs.tight_layout(rect=[0, 0, 1, 0.94])
fig_fs.savefig("figures/khi_viscous_final_state.png", dpi=150)
print("Saved → figures/khi_viscous_final_state.png")
plt.close(fig_fs)


# =============================================================================
#  §8  SIDE-BY-SIDE ANIMATION  (density)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

rho_norm = LogNorm(vmin=0.5, vmax=15.0, clip=True)

im1 = ax1.imshow(
    sl(result_A.states[0])[0].T,
    norm=rho_norm, cmap="viridis", origin="lower", extent=extent, aspect="equal",
)
im2 = ax2.imshow(
    sl(result_B.states[0])[0].T,
    norm=rho_norm, cmap="viridis", origin="lower", extent=extent, aspect="equal",
)

ax1.set_title(
    f"Case A:  Re$_b = {Re_A}$  $\\gg$  Re$_{{\\rm crit}}$\n"
    f"low viscosity  →  KHI develops",
    fontsize=11,
)
ax2.set_title(
    f"Case B:  Re$_b = {Re_B}$  $\\ll$  Re$_{{\\rm crit}}$\n"
    f"high viscosity  →  KHI suppressed",
    fontsize=11,
)
for ax in (ax1, ax2):
    ax.set_xlabel("$x$  (flow direction)")
    ax.axhline(y_center - Rs, color="w", ls="--", lw=0.5, alpha=0.4)
    ax.axhline(y_center + Rs, color="w", ls="--", lw=0.5, alpha=0.4)
ax1.set_ylabel("$y$  (transverse)")

time_txt = fig.suptitle(
    f"Viscous KHI  —  $\\delta = {delta:.0f}$,  Mach $= {Mb}$,  "
    f"$\\gamma = {gamma:.2f}$   |   $t = 0.000$",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 0.93, 0.91])


def update(frame):
    im1.set_data(sl(result_A.states[frame])[0].T)
    im2.set_data(sl(result_B.states[frame])[0].T)
    t_now = float(snap_times[frame])
    time_txt.set_text(
        f"Viscous KHI  —  $\\delta = {delta:.0f}$,  Mach $= {Mb}$,  "
        f"$\\gamma = {gamma:.2f}$   |   "
        f"$t = {t_now:.3f}$  "
        f"($t/\\tau_{{\\rm inv}} = {t_now/tau_KH_inv:.1f}$)"
    )
    return [im1, im2, time_txt]


print("\nRendering animation …")
ani = animation.FuncAnimation(fig, update, frames=n_snaps, blit=True, interval=80)
ani.save("figures/khi_viscous_critical_Re.gif", writer="pillow", fps=20, dpi=120)
plt.close(fig)
print("Saved → figures/khi_viscous_critical_Re.gif\n")

print("Done.")