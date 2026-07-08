"""Shared helper for the differentiability paper-figure drivers.

Each driver in this directory is a thin wrapper that runs the real example
script under ``examples/scripts/differentiability/`` in a subprocess (its own
directory as cwd, repo root on PYTHONPATH, an autocvd-selected GPU) and copies
the produced figure(s) into this area's ``figures/``. The heavy lifting and the
physics live in the example; the driver only wires it into ``--reproduce-paper``.
"""

# general
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples" / "scripts" / "differentiability"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def run_example_and_collect(script_name, figure_map):
    """Run ``examples/scripts/differentiability/<script_name>`` and copy figures.

    Args:
        script_name: Example filename (e.g. ``"eigen_initialization.py"``).
        figure_map: Mapping ``{generated_basename: dest_basename}`` copied from
            the example's ``figures/`` into this area's ``figures/``.
    """
    src = EXAMPLES / script_name
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {src} ...", flush=True)
    subprocess.run([sys.executable, str(src)], cwd=src.parent, env=env, check=True)
    FIG_DIR.mkdir(exist_ok=True)
    for generated, dest in figure_map.items():
        shutil.copy(src.parent / "figures" / generated, FIG_DIR / dest)
        print(f"copied {generated} -> figures/{dest}", flush=True)
