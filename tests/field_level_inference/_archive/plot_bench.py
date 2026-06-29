#!/usr/bin/env python
"""Plot the twin-benchmark convergence: terminal error + IC error vs wall-time
for the prolongation-free multigrid warm-start vs direct 64³ (same k-schedule)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("bench_mg_vs_direct.npz")
mg, direct = d["mg"], d["direct"]          # columns: cumtime, term_err, ic_err
sched = d["sched"]; gap = int(d["gap"]); k_true = int(d["k_true"])

# multigrid 32³->64³ transition: first row whose band k_cut > 16-gap.
# reconstruct per-row band by replaying the schedule lengths.
def transition_time(rec):
    i = 0
    for (kcut, steps) in sched:
        if kcut > 16 - gap:
            return rec[i, 0] if i < len(rec) else None
        i += steps
    return None

t_tr = transition_time(mg)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, col, name in [(ax[0], 1, "terminal error  (projection MSE)"),
                     (ax[1], 2, r"IC error  $\|v-v_{\rm true}\|/\|v_{\rm true}\|$")]:
    a.plot(direct[:, 0], direct[:, col], "-o", ms=3, color="#c0392b", label="direct 64³")
    a.plot(mg[:, 0], mg[:, col], "-o", ms=3, color="#2471b2",
           label="multigrid (32³ coarse → 64³)")
    if t_tr is not None:
        a.axvline(t_tr, ls="--", lw=1, color="#2471b2", alpha=0.6)
        a.text(t_tr, a.get_ylim()[1], " 32³→64³", color="#2471b2", va="top", fontsize=9)
    a.set_xlabel("cumulative optimisation wall-time  [s]")
    a.set_ylabel(name); a.set_yscale("log"); a.grid(alpha=0.3)
    a.legend(frameon=False)
fig.suptitle(f"Prolongation-free multigrid warm-start vs direct 64³  "
             f"(twin, |k_true|<{k_true}, Nyquist gap={gap})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("bench_mg_vs_direct.png", dpi=140)
print("wrote bench_mg_vs_direct.png")
print(f"  multigrid final: term={mg[-1,1]:.3e}  IC={mg[-1,2]:.3f}  @ {mg[-1,0]:.0f}s")
print(f"  direct    final: term={direct[-1,1]:.3e}  IC={direct[-1,2]:.3f}  @ {direct[-1,0]:.0f}s")
# time to reach the multigrid's final terminal error, for each method
tgt = mg[-1, 1]
def t_to(rec, thr):
    idx = np.where(rec[:, 1] <= thr)[0]
    return rec[idx[0], 0] if idx.size else None
print(f"  wall-time to reach term={tgt:.2e}:  mg={t_to(mg,tgt):.0f}s  "
      f"direct={t_to(direct,tgt)}")
