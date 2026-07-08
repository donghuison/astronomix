"""Field-level inference paper figure.

Regenerates ``field_level_inference.png`` by running the field-level inference
example (reverse-mode reconstruction of an initial velocity field from a target
column-density image).
"""

# testing
from _common import run_example_and_collect


def main():
    run_example_and_collect(
        "field_level_inference.py",
        {"field_level_inference.png": "field_level_inference.png"},
    )


if __name__ == "__main__":
    main()
