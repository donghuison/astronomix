"""3D circularly-polarized Alfvén-wave convergence figure (used as-is, double
precision).

This figure (``alfven_wave3D_dp_convergence.svg``) is produced by the existing
methods-paper benchmark in ``pytests/mhd/alfven_wave3D.py`` — an L1 convergence
sweep over N = 8..128 (double precision) for FV (JAX), FD (JAX) and FD
(Pallas), with the AthenaPK reference overlaid.  A copy of the validated figure
already lives in ``figures/``; running this script re-runs the (double
precision) benchmark and refreshes the copy.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/alfven_convergence.py
"""

# general
import os
import shutil
import subprocess
import sys
from pathlib import Path


# This script is a thin driver: the actual convergence figure is produced by
# the existing benchmark under ``pytests/mhd``. We resolve that source script,
# the figure it generates and the destination copy here relative to this file.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "pytests" / "mhd" / "alfven_wave3D.py"
GENERATED = SRC.parent / "figures" / "alfven_wave3D_dp_convergence.svg"
DEST = HERE / "figures" / "alfven_wave3D_dp_convergence.svg"


def main():
    """Re-run the double-precision 3D Alfvén convergence sweep and copy its
    figure into this directory.

    The heavy lifting is delegated to ``pytests/mhd/alfven_wave3D.py`` run in a
    subprocess (with the repo on ``PYTHONPATH`` so it picks up this worktree);
    the generated ``.svg`` is then copied next to the other paper figures.
    """
    # Pass the repo root on PYTHONPATH so the subprocess imports this worktree's
    # astronomix rather than the non-editable site-packages install.
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC} --convergence --dp (re-runs the 3D convergence sweep) ...")
    subprocess.run(
        [sys.executable, str(SRC), "--convergence", "--dp"],
        cwd=SRC.parent,
        env=env,
        check=True,
    )
    shutil.copy(GENERATED, DEST)
    print(f"copied {GENERATED} -> {DEST}")


if __name__ == "__main__":
    main()
