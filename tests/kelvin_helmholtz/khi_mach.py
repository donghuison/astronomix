# =============================================================================
#  Kelvin–Helmholtz Instability: Critical Mach Number Demonstration
#  Based on Mandelker et al. (2016), MNRAS 463, 3921  [arXiv:1610.03614]
#
#  Two simulations of a dense slab (stream) flowing through a dilute
#  background, one below and one above the critical Mach number:
#
#     M_crit = (1 + δ^{-1/3})^{3/2}       (Eq. 22)
#
#  Case A  (M_b < M_crit):  surface modes grow rapidly  → stream disrupts
#  Case B  (M_b > M_crit):  surface modes suppressed    → stream survives
# =============================================================================

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

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
    FORWARDS,
    HYBRID_HLLC,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)


# =============================================================================
#  §1  PHYSICAL PARAMETERS
# =============================================================================
#
#  We set up pressure equilibrium (P uniform everywhere) between a dense
#  cold stream (subscript s) and a hot dilute background (subscript b).
#
#  The density contrast δ = ρ_s/ρ_b sets M_crit.
#  The stream velocity V = M_b · c_b is measured in units of the background
#  sound speed (paper's convention: background at rest).
#
#  For δ = 10:   M_crit ≈ 1.77
#  ─────────────────────────────────────────────────────────────────────────

gamma = 5.0 / 3.0        # adiabatic index (both fluids)
delta = 10.0              # density contrast  ρ_s / ρ_b

rho_b = 1.0               # background density
rho_s = delta * rho_b     # stream density  (= 10)
P0    = 1.0               # uniform pressure  (pressure equilibrium)

c_b = float(jnp.sqrt(gamma * P0 / rho_b))   # ≈ 1.291
c_s = float(jnp.sqrt(gamma * P0 / rho_s))   # ≈ 0.408

# critical Mach number for surface-mode suppression  (Eq. 22)
M_crit = (1.0 + delta ** (-1.0 / 3.0)) ** 1.5   # ≈ 1.77

# ── two cases straddling M_crit ──────────────────────────────────────────
Mb_A = 1.0    # sub-critical   → surface modes UNSTABLE
Mb_B = 2.25   # super-critical → surface modes STABLE, body modes only

V_A = Mb_A * c_b     # stream velocity, Case A
V_B = Mb_B * c_b     # stream velocity, Case B

# total Mach number  M_tot = V / (c_s + c_b)   (Eq. 32: threshold for body modes)
Mtot_A = V_A / (c_s + c_b)
Mtot_B = V_B / (c_s + c_b)

print("=" * 65)
print(" KHI Critical Mach Number — Mandelker et al. (2016)")
print("=" * 65)
print(f"  γ = {gamma:.4f},   δ = {delta:.0f}")
print(f"  c_b = {c_b:.4f},   c_s = {c_s:.4f}")
print(f"  M_crit = {M_crit:.4f}")
print()
print(f"  Case A  Mb = {Mb_A:.2f}  < Mcrit  →  surface modes UNSTABLE")
print(f"          V = {V_A:.4f},  Mtot = {Mtot_A:.3f}  (body modes {'ON' if Mtot_A > 1 else 'off'})")
print()
print(f"  Case B  Mb = {Mb_B:.2f}  > Mcrit  →  surface modes STABLE")
print(f"          V = {V_B:.4f},  Mtot = {Mtot_B:.3f}  (body modes {'ON' if Mtot_B > 1 else 'off'})")
print("=" * 65)


# =============================================================================
#  §2  SIMULATION CONFIGURATION
# =============================================================================
box_size      = 1.0
num_cells     = 512       # ← reduce to 256 for a faster run
num_timesteps = 12000     # upper bound; CFL-adaptive stepping stops at t_end
t_end         = 1.5       # long enough for ~15 e-folds (Case A) vs ~2 (Case B)
    
solver_mode = FINITE_DIFFERENCE

config = SimulationConfig(
    solver_mode         = solver_mode,
    progress_bar        = True,
    dimensionality      = 2,
    box_size            = box_size,
    num_cells           = num_cells,
    num_timesteps       = num_timesteps,
    boundary_settings   = BoundarySettings(
        x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),   # flow dir
        y = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),   # transverse
    ),
    limiter        = DOUBLE_MINMOD,
    return_snapshots = True,
    num_snapshots  = 100,
    # does not apply for finite difference
    riemann_solver = HYBRID_HLLC,
)

helper_data          = get_helper_data(config)
params               = SimulationParams(t_end=t_end, C_cfl=0.4)
registered_variables = get_registered_variables(config)


# =============================================================================
#  §3  GRID, SLAB GEOMETRY, AND PERTURBATION
# =============================================================================
#
#  Slab (= stream cross-section) centred at y = 0.5
#     half-width  Rs = 0.1   →   slab spans  y ∈ [0.4, 0.6]
#
#  Interface smoothed with tanh on scale σ  (Eq. 48).
#
#  Perturbation: transverse velocity  v_y = A · sin(k·x)
#     5 wavelengths in the box  →  λ = 0.2 = 2·Rs
#     dimensionless wavenumber  K = k·Rs = π   (same as the paper's runs)
# ─────────────────────────────────────────────────────────────────────────

dx = box_size / num_cells
x  = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, num_cells)  # cell centres
y  = jnp.linspace(0.5 * dx, box_size - 0.5 * dx, num_cells)
X, Y = jnp.meshgrid(x, y, indexing="ij")     # X[i,j] = x_i,  Y[i,j] = y_j

y_center = 0.5
Rs       = 0.1           # slab half-width
sigma    = 0.008          # smoothing width  (σ/Δx ≈ 4 at 512 cells)

# perturbation
n_waves  = 5                                    # wavelengths in the box
k_pert   = 2.0 * jnp.pi * n_waves / box_size   # wavenumber
A_pert   = 0.01 * c_s                           # small: ~1 % of stream sound speed

t_sc = 2.0 * Rs / c_s          # stream sound-crossing time  (delay for body modes)

print(f"\n  Slab:  y ∈ [{y_center-Rs:.2f}, {y_center+Rs:.2f}],  Rs = {Rs}")
print(f"  Smoothing:  σ = {sigma},  σ/Δx = {sigma/dx:.1f}")
print(f"  Perturbation:  {n_waves} waves,  λ = {box_size/n_waves:.3f},  K = k·Rs = {float(k_pert*Rs):.4f}")
print(f"  Sound-crossing time (stream):  t_sc = {t_sc:.4f}")
print(f"  Simulation end:  t_end = {t_end}\n")


# ── smoothed slab profile (Eq. 48 in the paper) ─────────────────────────
def slab_profile(f_b, f_s):
    """Tanh transition from f_b (background) to f_s (stream)."""
    return f_b + 0.5 * (f_s - f_b) * (
        1.0 + jnp.tanh((Rs - jnp.abs(Y - y_center)) / sigma)
    )


# ── build initial primitive state for a given Mach number ────────────────
def create_initial_state(Mb):
    V_stream = Mb * c_b

    rho = slab_profile(rho_b, rho_s)           # density
    vx  = slab_profile(0.0,   V_stream)         # stream flows in +x
    vy  = A_pert * jnp.sin(k_pert * X)          # transverse kick (non-eigenmode)
    p   = P0 * jnp.ones_like(X)                 # uniform pressure

    return construct_primitive_state(
        config              = config,
        registered_variables = registered_variables,
        density             = rho,
        velocity_x          = vx,
        velocity_y          = vy,
        gas_pressure        = p,
    )


# =============================================================================
#  §4  CREATE INITIAL STATES & FINALIZE
# =============================================================================
state_A = create_initial_state(Mb_A)
state_B = create_initial_state(Mb_B)
config  = finalize_config(config, state_A.shape)   # both states have same shape


# =============================================================================
#  §5  RUN SIMULATIONS
# =============================================================================
print("━" * 65)
print(f"  Running Case A   Mb = {Mb_A}  (sub-critical) …")
print("━" * 65)
t0 = timer()
result_A = time_integration(state_A, config, params, registered_variables)
print(f"  ✓ Case A finished in {timer()-t0:.1f} s\n")

print("━" * 65)
print(f"  Running Case B   Mb = {Mb_B}  (super-critical) …")
print("━" * 65)
t0 = timer()
result_B = time_integration(state_B, config, params, registered_variables)
print(f"  ✓ Case B finished in {timer()-t0:.1f} s\n")


# =============================================================================
#  §6  GROWTH-RATE DIAGNOSTIC
# =============================================================================
#
#  Track the volume-averaged pressure perturbation |P − P0| inside the
#  slab region as a proxy for the instability amplitude (Sec. 3.2).
#  Expect:
#    Case A — exponential growth starting at t ∼ t_λ = λ/c_b  (tiny)
#    Case B — no growth until t ∼ t_sc, then much slower exponential
# ─────────────────────────────────────────────────────────────────────────
n_snaps = min(len(result_A.states), len(result_B.states))
snap_times = jnp.linspace(0, t_end, n_snaps)

# pressure is stored at index 4 in the primitive state (after ρ, vx, vy, E)
# (check your astronomix version – adapt the index if needed)
p_idx = registered_variables.pressure_index

# slab region in y  (indices)
j_slab_lo = int((y_center - 1.5 * Rs) * num_cells)
j_slab_hi = int((y_center + 1.5 * Rs) * num_cells)

amp_A, amp_B = [], []
for i in range(n_snaps):
    dP_A = jnp.abs(result_A.states[i][p_idx, :, j_slab_lo:j_slab_hi] - P0)
    dP_B = jnp.abs(result_B.states[i][p_idx, :, j_slab_lo:j_slab_hi] - P0)
    amp_A.append(float(jnp.mean(dP_A)))
    amp_B.append(float(jnp.mean(dP_B)))

amp_A = jnp.array(amp_A)
amp_B = jnp.array(amp_B)

# ── plot ──
fig_gr, ax_gr = plt.subplots(figsize=(8, 5))
ax_gr.semilogy(snap_times, amp_A, "C0-",  lw=2,
               label=f"Case A: $M_b={Mb_A}$ (sub-crit)")
ax_gr.semilogy(snap_times, amp_B, "C3--", lw=2,
               label=f"Case B: $M_b={Mb_B}$ (super-crit)")

ax_gr.axvline(t_sc, color="C3", ls=":", lw=1, alpha=0.6)
ax_gr.text(t_sc * 1.03, ax_gr.get_ylim()[0] * 3, "$t_{\\rm sc}$", color="C3",
           fontsize=11)

ax_gr.set_xlabel("time", fontsize=12)
ax_gr.set_ylabel(r"$\langle\,|\,P - P_0\,|\,\rangle$   (slab region)", fontsize=12)
ax_gr.set_title(
    f"KHI growth rate  —  $\\delta={delta:.0f}$,  "
    f"$M_{{\\rm crit}}={M_crit:.2f}$",
    fontsize=13,
)
ax_gr.legend(fontsize=11)
ax_gr.grid(True, which="both", alpha=0.3)
fig_gr.tight_layout()
fig_gr.savefig("figures/khi_growth_rate.png", dpi=150)
print("Saved → figures/khi_growth_rate.png")
plt.close(fig_gr)


# =============================================================================
#  §7  FINAL-STATE COMPARISON  (density, v_y, pressure perturbation)
# =============================================================================
fig_fs, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)

y_lo, y_hi = 0.15, 0.85
extent = [0, box_size, y_lo, y_hi]
jlo = int(y_lo * num_cells)
jhi = int(y_hi * num_cells)

# handy slicing
def sl(state):
    return state[:, :, jlo:jhi]

labels_top = [
    f"Case A  $M_b={Mb_A}$  (sub-crit)",
    f"Case A  $M_b={Mb_A}$",
    f"Case A  $M_b={Mb_A}$",
]
labels_bot = [
    f"Case B  $M_b={Mb_B}$  (super-crit)",
    f"Case B  $M_b={Mb_B}$",
    f"Case B  $M_b={Mb_B}$",
]

sA = result_A.states[-1]
sB = result_B.states[-1]

# ── density ──
rho_norm = LogNorm(vmin=0.5, vmax=15, clip=True)
axes[0, 0].imshow(sl(sA)[0].T, norm=rho_norm, cmap="viridis", origin="lower", extent=extent)
axes[1, 0].imshow(sl(sB)[0].T, norm=rho_norm, cmap="viridis", origin="lower", extent=extent)
axes[0, 0].set_title("Density  ρ", fontsize=12)

# ── transverse velocity ──
vy_lim = float(jnp.max(jnp.abs(sl(sA)[2])))
vy_lim = max(vy_lim, float(jnp.max(jnp.abs(sl(sB)[2]))))
axes[0, 1].imshow(sl(sA)[2].T, cmap="RdBu_r", origin="lower", extent=extent,
                  vmin=-vy_lim, vmax=vy_lim)
axes[1, 1].imshow(sl(sB)[2].T, cmap="RdBu_r", origin="lower", extent=extent,
                  vmin=-vy_lim, vmax=vy_lim)
axes[0, 1].set_title("Transverse velocity  $v_y$", fontsize=12)

# ── pressure perturbation ──
dP_A = sl(sA)[p_idx] - P0
dP_B = sl(sB)[p_idx] - P0
dp_lim = float(jnp.max(jnp.abs(dP_A)))
dp_lim = max(dp_lim, float(jnp.max(jnp.abs(dP_B))), 1e-6)
axes[0, 2].imshow(dP_A.T, cmap="RdBu_r", origin="lower", extent=extent,
                  vmin=-dp_lim, vmax=dp_lim)
axes[1, 2].imshow(dP_B.T, cmap="RdBu_r", origin="lower", extent=extent,
                  vmin=-dp_lim, vmax=dp_lim)
axes[0, 2].set_title("Pressure perturbation  $P - P_0$", fontsize=12)

for ax in axes[0, :]:
    ax.set_aspect("equal")
for ax in axes[1, :]:
    ax.set_aspect("equal")
    ax.set_xlabel("x  (flow)", fontsize=11)
axes[0, 0].set_ylabel(f"Case A   $M_b={Mb_A}$\ny", fontsize=11)
axes[1, 0].set_ylabel(f"Case B   $M_b={Mb_B}$\ny", fontsize=11)

# slab boundaries
for row in axes:
    for ax in row:
        ax.axhline(y_center - Rs, color="w", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(y_center + Rs, color="w", ls="--", lw=0.6, alpha=0.5)

fig_fs.suptitle(
    f"Final state at t = {t_end}   —   $\\delta={delta:.0f}$,  "
    f"$M_{{\\rm crit}}={M_crit:.2f}$,  $\\gamma={gamma:.2f}$",
    fontsize=14, y=0.98,
)
fig_fs.tight_layout(rect=[0, 0, 1, 0.95])
fig_fs.savefig("figures/khi_final_state.png", dpi=150)
print("Saved → figures/khi_final_state.png")
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
    f"Case A:  $M_b = {Mb_A}$  <  $M_{{\\rm crit}} = {M_crit:.2f}$\n"
    f"surface modes unstable",
    fontsize=11,
)
ax2.set_title(
    f"Case B:  $M_b = {Mb_B}$  >  $M_{{\\rm crit}} = {M_crit:.2f}$\n"
    f"surface modes suppressed",
    fontsize=11,
)
for ax in (ax1, ax2):
    ax.set_xlabel("x  (flow direction)")
    ax.axhline(y_center - Rs, color="w", ls="--", lw=0.5, alpha=0.4)
    ax.axhline(y_center + Rs, color="w", ls="--", lw=0.5, alpha=0.4)
ax1.set_ylabel("y  (transverse)")

time_txt = fig.suptitle(
    f"KHI in cold streams  —  $\\delta = {delta:.0f}$,  $\\gamma = {gamma:.2f}$"
    f"   |   t = 0.000",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 0.93, 0.91])


def update(frame):
    im1.set_data(sl(result_A.states[frame])[0].T)
    im2.set_data(sl(result_B.states[frame])[0].T)
    t_now = float(snap_times[frame])
    time_txt.set_text(
        f"KHI in cold streams  —  $\\delta = {delta:.0f}$,  "
        f"$\\gamma = {gamma:.2f}$   |   t = {t_now:.3f}"
    )
    return [im1, im2, time_txt]


print("\nRendering animation …")
ani = animation.FuncAnimation(fig, update, frames=n_snaps, blit=True, interval=80)
ani.save("figures/khi_critical_mach.gif", writer="pillow", fps=20, dpi=120)
plt.close(fig)
print("Saved → figures/khi_critical_mach.gif\n")

print("Done.")