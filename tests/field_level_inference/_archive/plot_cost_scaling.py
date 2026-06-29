#!/usr/bin/env python
"""Cost-scaling validation for the paper: measured per-step BACKWARD (gradient)
wall-time vs resolution N, against the theoretical N⁴ law (N³ cells × N CFL steps).
Numbers are measured Pallas MHD value_and_grad step times on this machine."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# measured backward (value_and_grad) step times [s] for the MHD logo/twin loss
N = np.array([32, 64, 128], dtype=float)
t = np.array([6.0, 103.0, 1570.0])          # 32³: mg coarse stage; 64³: naive; 128³: probe

# N⁴ reference anchored at 64³
ref = t[1] * (N / 64.0) ** 4

fig, ax = plt.subplots(figsize=(6.6, 5.0))
ax.loglog(N, t, "o", ms=9, color="#1f6fb2", label="measured backward step", zorder=3)
ax.loglog(N, ref, "--", color="#888888", lw=1.6, label=r"$N^4$  (cells $N^3\times$ CFL steps $N$)")
for ni, ti in zip(N, t):
    ax.annotate(f"{ti:.0f}s", (ni, ti), textcoords="offset points", xytext=(8, -4), fontsize=10)
ax.set_xlabel("resolution  N  (cells per dimension)")
ax.set_ylabel("backward (gradient) wall-time per step  [s]")
ax.set_xticks(N); ax.set_xticklabels([f"{int(n)}³" for n in N])
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, fontsize=10)
ax.set_title("Differentiable MHD step cost vs resolution", fontsize=12)
# annotate measured doubling ratios
for i in range(len(N) - 1):
    r = t[i + 1] / t[i]
    ax.annotate(f"×{r:.0f}", (np.sqrt(N[i] * N[i + 1]), np.sqrt(t[i] * t[i + 1])),
                color="#1f6fb2", fontsize=11, ha="center")
fig.tight_layout()
fig.savefig("cost_scaling.png", dpi=140)
print("wrote cost_scaling.png")
print("  measured per-doubling ratios:", [f"{t[i+1]/t[i]:.1f}x" for i in range(len(N)-1)],
      " (N^4 predicts 16x)")
