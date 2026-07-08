"""KHI initial-condition reconstruction paper figures.

The paper's shooting-study figures (``study_2x3_N256``, ``recon_2x5_N256``,
``recon_4x3_N256``) come from the full single-vs-multiple-shooting campaign in
the ``kh_recon`` example (N=256, both horizons, 16 cold inits each). That
campaign is expensive, so it is NOT run automatically here: run it once with

    python examples/scripts/differentiability/kh_recon.py run --horizon 20 --init <i>
    python examples/scripts/differentiability/kh_recon.py run --horizon 60 --init <i>

for i in 0..15 (fan across GPUs), then this driver copies the ``plot``/``recon``
figures. With the campaign cached, it regenerates and copies the figures; without
it, the underlying ``plot``/``recon`` step raises so the missing campaign is
obvious rather than silently skipped.
"""

# general
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "examples" / "scripts" / "differentiability" / "kh_recon.py"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def main():
    """Regenerate the shooting-study figures from the cached campaign."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    FIG_DIR.mkdir(exist_ok=True)
    # 2x3 convergence summary + the 2x5 / 4x3 reconstruction montages
    subprocess.run([sys.executable, str(SRC), "plot", "--horizons", "20", "60"],
                   cwd=SRC.parent, env=env, check=True)
    subprocess.run([sys.executable, str(SRC), "recon", "--horizons", "20", "60"],
                   cwd=SRC.parent, env=env, check=True)
    for name in ("study_2x3_N256.png", "recon_2x5_N256.png", "recon_4x3_N256.png"):
        generated = SRC.parent / "figures" / name
        if generated.exists():
            shutil.copy(generated, FIG_DIR / name)
            print(f"copied {name}", flush=True)


if __name__ == "__main__":
    main()
