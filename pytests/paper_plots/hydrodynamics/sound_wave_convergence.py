"""3D linear sound-wave convergence figure (used as-is in the paper).

Refreshes ``figures/sound_wave3D_convergence.svg`` — an L1 convergence sweep
over N = 8..128 for FV (JAX), FD (JAX) and FD (Pallas). The figure is produced
by the existing methods-paper benchmark in
``pytests/hydrodynamics/sound_wave3D.py``; a copy of the validated figure
already lives in ``figures/``, and running this script re-runs that benchmark
and refreshes the copy.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/sound_wave_convergence.py
"""

# general
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "pytests" / "hydrodynamics" / "sound_wave3D.py"
GENERATED = REPO / "pytests" / "hydrodynamics" / "figures" / "sound_wave3D_convergence.svg"
DEST = HERE / "figures" / "sound_wave3D_convergence.svg"


def main():
    """Re-run the 3D sound-wave convergence sweep and copy its figure.

    The benchmark is launched as a subprocess (``sound_wave3D.py
    --convergence``) with cwd set to the source directory, then the generated
    SVG is copied into this area's ``figures/``.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC} --convergence (this re-runs the 3D convergence sweep) ...")
    subprocess.run(
        [sys.executable, str(SRC), "--convergence"],
        cwd=SRC.parent,
        env=env,
        check=True,
    )
    shutil.copy(GENERATED, DEST)
    print(f"copied {GENERATED} -> {DEST}")


if __name__ == "__main__":
    main()
