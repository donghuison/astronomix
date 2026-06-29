#!/usr/bin/env python
"""Money plot: terminal logo-matching loss (64³ projection MSE) vs optimisation
wall-time for naive / k-windowing / prolongation-free-multigrid warm start."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
d = dict(np.load("bench_money.npz"))
sched = d["sched"]; coarse_cut = int(d["coarse_cut"])
# prefer the equal-wall-time extended multigrid run if present
if os.path.exists("bench_money_mgext.npz"):
    dm = np.load("bench_money_mgext.npz")
    d["multigrid"] = dm["multigrid"]; sched = dm["sched"]
    print("using extended multigrid run")
STYLE = {
    "naive":       ("naive  (all modes, 64³)",            "#888888", "-o"),
    "k_windowing": ("k-windowing  (8→16→32, 64³)",   "#c0392b", "-o"),
    "multigrid":   ("multigrid + k-windowing  (32³→64³)", "#1f8a4c", "-o"),
}

# multigrid 32³->64³ transition wall-time: end of the last coarse band.
coarse_steps = sum(s for (kc, s) in sched if kc <= coarse_cut)
mg = d["multigrid"]
t_tr = mg[min(coarse_steps, len(mg)) - 1, 0] if coarse_steps else None

fig, ax = plt.subplots(figsize=(8.2, 5.6))
for key, (lab, col, ls) in STYLE.items():
    if key not in d:
        continue
    r = d[key]
    ax.plot(r[:, 0], r[:, 1], ls, color=col, ms=4, lw=2, label=lab)
if t_tr is not None:
    ax.axvline(t_tr, ls="--", lw=1, color="#1f8a4c", alpha=0.6)
    ax.text(t_tr, ax.get_ylim()[1], " 32³→64³", color="#1f8a4c", va="top", fontsize=9)
ax.set_xlabel("cumulative optimisation wall-time  [s]")
ax.set_ylabel("terminal loss   (64³ logo-projection MSE)")
ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, fontsize=10)
ax.set_title("Logo reconstruction: convergence vs compute", fontsize=12)
fig.tight_layout()
fig.savefig("money.png", dpi=140)
print("wrote money.png")

# speedup: wall-time each method needs to reach the naive method's final loss
def t_to(r, thr):
    idx = np.where(r[:, 1] <= thr)[0]
    return r[idx[0], 0] if idx.size else None
naive_final = d["naive"][-1, 1]
print(f"  naive final loss64 = {naive_final:.3e} @ {d['naive'][-1,0]:.0f}s")
for key in ("k_windowing", "multigrid"):
    if key in d:
        t = t_to(d[key], naive_final)
        print(f"  {key:12s}: reaches naive-final loss in {t}s "
              f"(final {d[key][-1,1]:.3e} @ {d[key][-1,0]:.0f}s)")
