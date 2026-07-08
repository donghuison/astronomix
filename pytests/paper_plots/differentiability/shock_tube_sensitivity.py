"""Shock-tube sensitivity paper figure.

Regenerates ``shock_tube_sensitivity_step_sweep`` (AD vs finite-difference
gradients of a Sod shock-tube cost over a step-size sweep) by running the
shock-tube sensitivity example.
"""

# testing
from _common import run_example_and_collect


def main():
    run_example_and_collect(
        "shock_tube_sensitivity.py",
        {"shock_tube_sensitivity_step_sweep.svg": "shock_tube_sensitivity_step_sweep.svg"},
    )


if __name__ == "__main__":
    main()
