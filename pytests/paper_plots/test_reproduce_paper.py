"""Regenerate the methods-paper figures.

Every generator script under ``pytests/paper_plots/<area>/`` is run as a
subprocess when the suite is invoked with ``--reproduce-paper``::

    pytest pytests/paper_plots --reproduce-paper

Without the flag the cases are skipped, so a normal ``pytest`` run stays fast.
Each generator writes into its area's ``figures/`` (and caches to ``data/``);
those outputs are gitignored. Helper modules (``_common.py`` etc.) are excluded.
"""

# general
import os
import subprocess
import sys
from pathlib import Path

# testing
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

GENERATORS = sorted(
    p for p in HERE.rglob("*.py")
    if p.name != Path(__file__).name and not p.name.startswith("_")
)


@pytest.mark.parametrize(
    "script", GENERATORS, ids=lambda p: str(p.relative_to(HERE))
)
def test_reproduce_paper_figure(script, request):
    """Run one paper-figure generator as a subprocess.

    Skipped unless ``--reproduce-paper`` is passed, so a normal ``pytest`` run
    stays fast. Each generator is executed with the repository root on
    PYTHONPATH and its own directory as cwd.

    Args:
        script: The generator script path (one parametrized case per figure).
        request: The pytest request, used to read the ``--reproduce-paper`` flag.
    """
    if not request.config.getoption("--reproduce-paper"):
        pytest.skip("pass --reproduce-paper to regenerate the paper figures")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    subprocess.run(
        [sys.executable, str(script)], cwd=script.parent, check=True, env=env
    )
