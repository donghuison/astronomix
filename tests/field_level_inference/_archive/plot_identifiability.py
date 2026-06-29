#!/usr/bin/env python
"""Twin identifiability summary: can the initial velocity be recovered from a
density observable?  Plots corr(v,v_true) and the Helmholtz-split IC error vs
optimisation step for the density observable at three horizons (t_mult=0.5/1/2)
and the full-state control (t_mult=1).  rec columns: cum,L,ic_tot,ic_comp,ic_sol,corr."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("bench_ff64_tmult05.npz",   "density  t=0.5 t_c",  "#9ecae1", "-"),
    ("bench_ff64_split.npz",     "density  t=1.0 t_c",  "#4495c8", "-"),
    ("bench_ff64_tmult2.npz",    "density  t=2.0 t_c",  "#08519c", "-"),
    ("bench_ff64_fullstate.npz", "full state  t=1.0 t_c", "#c0392b", "-"),
]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for fn, lab, col, ls in RUNS:
    try:
        rec = np.load(fn)["mg"]
    except FileNotFoundError:
        print(f"skip missing {fn}"); continue
    step = np.arange(1, len(rec) + 1)
    ax[0].plot(step, rec[:, 5], ls, color=col, lw=2, label=lab)         # corr
    ax[1].plot(step, rec[:, 2], ls, color=col, lw=2, label=lab)         # IC total

ax[0].axhline(0, color="k", lw=0.8, alpha=0.5)
ax[0].set_ylabel(r"correlation  $\langle v,\,v_{\rm true}\rangle / \|v\|\|v_{\rm true}\|$")
ax[0].set_title("velocity recovery (1 = perfect, 0 = unrelated)")
ax[1].axhline(1, color="k", lw=0.8, alpha=0.5)
ax[1].set_ylabel(r"IC error  $\|v-v_{\rm true}\| / \|v_{\rm true}\|$")
ax[1].set_title("initial-condition error (1 = no better than v=0)")
for a in ax:
    a.set_xlabel("optimisation step"); a.grid(alpha=0.3); a.legend(frameon=False, fontsize=9)
fig.suptitle("Field-level inference twin: is the initial velocity identifiable from density?",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("identifiability.png", dpi=140)
print("wrote identifiability.png")
for fn, lab, _, _ in RUNS:
    try:
        r = np.load(fn)["mg"][-1]
    except FileNotFoundError:
        continue
    print(f"  {lab:22s} final: corr={r[5]:+.3f}  IC={r[2]:.3f}  "
          f"(comp={r[3]:.3f} sol={r[4]:.3f})  L={r[1]:.2e}")
