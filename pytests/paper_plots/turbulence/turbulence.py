"""Driven-turbulence paper figures.

Regenerates the turbulence spectra comparison and the final-density slices by
running the driven-MHD-turbulence example (a resolution sweep with
Ornstein-Uhlenbeck forcing). Copies the produced figures into this area's
``figures/``.
"""

# general
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "examples" / "scripts" / "forward" / "mhd" / "turbulence.py"
FIG_DIR = Path(__file__).resolve().parent / "figures"

# the example sweeps these resolutions (paper: [32, 64, 128, 256, 512])
RESOLUTIONS = [32, 64]


def main():
    """Run the turbulence example and copy its paper figures."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC} ...", flush=True)
    subprocess.run([sys.executable, str(SRC)], cwd=SRC.parent, env=env, check=True)
    FIG_DIR.mkdir(exist_ok=True)
    figures = ["turbulence_spectra_comparison.png"]
    figures += [f"turbulence_final_density_N{n}.png" for n in RESOLUTIONS]
    for name in figures:
        shutil.copy(SRC.parent / "figures" / name, FIG_DIR / name)
        print(f"copied {name}", flush=True)


if __name__ == "__main__":
    main()
