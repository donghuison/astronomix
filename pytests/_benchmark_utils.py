"""
Shared benchmark machinery for the methods-paper 3D linear-wave tests.

A "benchmark" here is the combination of:
- a problem-specific initial-state factory ``setup_fn`` and matching
  analytic-state factory ``analytic_fn``,
- one or more :class:`BenchmarkSpec` rows (a ``SimulationConfig`` template
  plus a label and CFL number),
- a sweep over grid resolutions.

The convergence/runtime pipeline and the strong-scaling+memory pipeline are
shared so that every physics-module benchmark in this paper produces the
same plot suite (see ``project_methods_paper_benchmarks`` memory).
"""

# general
import os
import re

# typing
from typing import Callable, NamedTuple, Optional, Sequence

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# astronomix constants
from astronomix.option_classes.simulation_config import (
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    SnapshotSettings,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables


class BenchmarkSpec(NamedTuple):
    """A single curve in a benchmark sweep."""

    #: Human-readable legend label, e.g. "FD (Pallas)".
    label: str
    #: SimulationConfig with everything set except ``num_cells`` (injected
    #: per-N inside the sweep).
    base_config: SimulationConfig
    #: CFL number for this configuration.
    cfl: float


def grid_shape_3d(N: int) -> StaticIntVector:
    """Standard methods-paper grid shape: (2N, N, N)."""
    return StaticIntVector(2 * N, N, N)


def _slug(text: str) -> str:
    """Lowercase, alphanumeric-and-underscore version of ``text``."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _ensure_dirs(*paths: str) -> None:
    """Create every given output directory (no error if it already exists)."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _build_sharding(num_gpus: int):
    """Build the multi-device sharding for a scaling run.

    Returns ``None`` for a single-GPU run (no sharding needed); otherwise a
    ``NamedSharding`` that splits the domain across ``num_gpus`` devices along
    the x-axis while leaving the variable, y and z axes replicated.
    """
    if num_gpus <= 1:
        return None
    mesh = jax.make_mesh((1, num_gpus, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
    return jax.NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def _ensure_snapshot_config(config: SimulationConfig) -> SimulationConfig:
    """Force the flags the helpers need so callers can supply a minimal config."""
    updates = {}
    if not config.return_snapshots:
        updates["return_snapshots"] = True
    if not config.print_elapsed_time:
        updates["print_elapsed_time"] = True
    if not config.memory_analysis:
        updates["memory_analysis"] = True
    if not config.snapshot_settings.return_final_state:
        updates["snapshot_settings"] = config.snapshot_settings._replace(
            return_final_state=True
        )
    return config._replace(**updates) if updates else config


def _run_simulation(spec: BenchmarkSpec, N: int, setup_fn: Callable, num_gpus: int = 1):
    """Build the per-N config, set up the IC, optionally shard, integrate.

    Returns ``(result, config, params, registered_variables, helper_data)``.
    """
    # Clear JIT/compilation caches between successive runs. Reusing the same
    # cached compile across (sharded, unsharded) inputs surfaces as
    # ``AttributeError: 'UnspecifiedValue' object has no attribute
    # addressable_devices_indices_map`` when the array sharding diverges
    # from what the cached trace expects.
    jax.clear_caches()

    base = _ensure_snapshot_config(spec.base_config)
    base = base._replace(num_cells=grid_shape_3d(N))
    initial_state, config, params = setup_fn(base, SimulationParams(C_cfl=spec.cfl))
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    sharding = _build_sharding(num_gpus)
    if sharding is not None:
        initial_state = jax.device_put(initial_state, sharding)

    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    # Make sure compute has finished before we read runtime/memory back.
    if hasattr(result, "final_state") and result.final_state is not None:
        result.final_state.block_until_ready()
    return result, config, params, registered_variables, helper_data


def _mean_l1_error(final_state, true_state, indices: Sequence[int]) -> float:
    """Mean over the requested variables of their per-variable mean L1 error."""
    per_variable_errors = [
        jnp.mean(jnp.abs(final_state[i] - true_state[i])) for i in indices
    ]
    return float(jnp.stack(per_variable_errors).mean())


def _format_xaxis(ax, N_values):
    """Put the resolution axis on a log scale with one labelled tick per N.

    The default log locator would sprinkle unlabelled minor ticks between the
    sampled resolutions; pinning the major ticks to exactly ``N_values`` keeps
    the axis readable for the small, hand-picked resolution list.
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(N_values))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(n) for n in N_values]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def run_convergence_and_runtime(
    benchmarks: Sequence[BenchmarkSpec],
    N_values: Sequence[int],
    setup_fn: Callable,
    analytic_fn: Callable,
    error_var_indices_fn: Callable,
    *,
    name: str,
    title: str,
    data_dir: str,
    figure_dir: str,
    athenapk_npz: Optional[str] = None,
) -> dict:
    """Run the convergence + runtime sweep for every benchmark.

    Produces two figures:
      * ``{figure_dir}/{name}_convergence.svg`` — L1 error vs N
      * ``{figure_dir}/{name}_runtime.svg``     — error/runtime, runtime/N,
        time-per-iteration/N

    and one NPZ per benchmark in ``{data_dir}``. Returns a dict keyed by
    benchmark label with the captured arrays.
    """
    _ensure_dirs(data_dir, figure_dir)

    results = {}
    for spec in benchmarks:
        results[spec.label] = dict(N=[], l1=[], runtime=[], iterations=[])

    fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 6))
    fig_rt, ax_rt = plt.subplots(3, 1, figsize=(8, 12))

    # -------------------------------------------------------------
    # ====== ↓ Run the sweep and collect per-benchmark curves ↓ ===
    # -------------------------------------------------------------

    # For every benchmark and resolution we integrate the wave, measure the L1
    # error against the analytic solution plus the runtime, plot the curves and
    # persist the raw arrays to an NPZ so the figures can be regenerated later.
    for spec in benchmarks:
        for N in N_values:
            result, config, params, registered_variables, helper_data = _run_simulation(
                spec,
                N,
                setup_fn,
                num_gpus=1,
            )

            indices = error_var_indices_fn(registered_variables)
            true_state = analytic_fn(config, registered_variables, params, helper_data)
            err = _mean_l1_error(result.final_state, true_state, indices)
            runtime = float(result.runtime)
            iters = int(result.num_iterations)

            results[spec.label]["N"].append(N)
            results[spec.label]["l1"].append(err)
            results[spec.label]["runtime"].append(runtime)
            results[spec.label]["iterations"].append(iters)

            print(f"[{name}] {spec.label} N={N}: L1={err:.3e}, runtime={runtime:.2f}s, iters={iters}")

        ax_err.loglog(
            N_values,
            results[spec.label]["l1"],
            marker="o",
            linewidth=2,
            label=spec.label,
        )
        ax_rt[0].loglog(
            results[spec.label]["runtime"],
            results[spec.label]["l1"],
            marker="o",
            linewidth=2,
            label=spec.label,
        )
        ax_rt[1].loglog(
            N_values,
            results[spec.label]["runtime"],
            marker="o",
            linewidth=2,
            label=spec.label,
        )
        time_per_iter = [
            rt / nit for rt, nit in zip(
                results[spec.label]["runtime"], results[spec.label]["iterations"]
            )
        ]
        ax_rt[2].loglog(
            N_values,
            time_per_iter,
            marker="o",
            linewidth=2,
            label=spec.label,
        )

        np.savez(
            os.path.join(data_dir, f"{name}_convergence_{_slug(spec.label)}.npz"),
            N_values=np.array(N_values),
            l1_errors=np.array(results[spec.label]["l1"]),
            runtimes=np.array(results[spec.label]["runtime"]),
            iterations=np.array(results[spec.label]["iterations"]),
        )

    # -------------------------------------------------------------
    # ====== ↑ Run the sweep and collect per-benchmark curves ↑ ===
    # -------------------------------------------------------------

    # Overlay the reference AthenaPK curve (when provided) for direct comparison.
    if athenapk_npz is not None and os.path.exists(athenapk_npz):
        athena = np.load(athenapk_npz)
        ax_err.loglog(
            athena["N_values"],
            athena["l1_errors"],
            marker="s",
            linewidth=2,
            label="AthenaPK",
        )
        ax_rt[0].loglog(
            athena["runtimes"],
            athena["l1_errors"],
            marker="s",
            linewidth=2,
            label="AthenaPK",
        )
        ax_rt[1].loglog(
            athena["N_values"],
            athena["runtimes"],
            marker="s",
            linewidth=2,
            label="AthenaPK",
        )
        ax_rt[2].loglog(
            athena["N_values"],
            [t / i for t, i in zip(athena["runtimes"], athena["iterations"])],
            marker="s",
            linewidth=2,
            label="AthenaPK",
        )

    # Reference convergence slopes: the FV scheme is expected to approach
    # second order and the FD scheme fifth order, so anchor both guide lines
    # at the coarsest-grid error for visual comparison.
    N_arr = np.array(N_values, dtype=float)
    ref2 = (N_arr / N_arr[0]) ** -2.0
    ref5 = (N_arr / N_arr[0]) ** -5.0
    max_err_start = max(r["l1"][0] for r in results.values() if r["l1"])
    min_err_start = min(r["l1"][0] for r in results.values() if r["l1"])
    ax_err.loglog(
        N_arr,
        max_err_start * ref2,
        "k--",
        alpha=0.7,
        label="$O(N^{-2})$ reference",
    )
    ax_err.loglog(
        N_arr,
        min_err_start * ref5,
        "k:",
        alpha=0.7,
        label="$O(N^{-5})$ reference",
    )

    # -------------------------------------------------------------
    # ============= ↓ Style and write the figures ↓ ===============
    # -------------------------------------------------------------

    ax_err.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_err.set_ylabel("Average $L_1$ error", fontsize=12)
    ax_err.set_title(f"{title}: convergence", fontsize=14)
    _format_xaxis(ax_err, N_values)
    ax_err.legend(loc="lower left", fontsize=9)
    ax_err.grid(True, which="major", ls="-", alpha=0.2)
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(figure_dir, f"{name}_convergence.svg"))

    ax_rt[0].set_xlabel("Runtime (s)", fontsize=12)
    ax_rt[0].set_ylabel("Average $L_1$ error", fontsize=12)
    ax_rt[0].set_title(f"{title}: error vs runtime", fontsize=13)
    ax_rt[1].set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_rt[1].set_ylabel("Runtime (s)", fontsize=12)
    ax_rt[1].set_title("Runtime vs N", fontsize=13)
    ax_rt[2].set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_rt[2].set_ylabel("Time per iteration (s)", fontsize=12)
    ax_rt[2].set_title("Time per iteration vs N", fontsize=13)
    for ax in (ax_rt[1], ax_rt[2]):
        _format_xaxis(ax, N_values)
    for ax in ax_rt:
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="major", ls="-", alpha=0.2)
    fig_rt.tight_layout()
    fig_rt.savefig(os.path.join(figure_dir, f"{name}_runtime.svg"))

    # -------------------------------------------------------------
    # ============= ↑ Style and write the figures ↑ ===============
    # -------------------------------------------------------------

    return results


def run_strong_scaling(
    benchmarks: Sequence[BenchmarkSpec],
    N_values: Sequence[int],
    setup_fn: Callable,
    *,
    num_gpus: int,
    name: str,
    title: str,
    data_dir: str,
    figure_dir: str,
) -> dict:
    """Run a 1-GPU vs ``num_gpus``-GPU sweep for every benchmark.

    Produces ``{figure_dir}/{name}_strong_scaling.svg`` containing
    runtime, speedup, temporary memory and argument memory panels, plus a
    single NPZ ``{data_dir}/{name}_strong_scaling.npz`` with all raw bytes.
    Returns the captured dict keyed by benchmark label.
    """
    _ensure_dirs(data_dir, figure_dir)

    available = jax.local_device_count()
    if available < num_gpus:
        raise RuntimeError(
            f"Strong scaling needs {num_gpus} GPUs but only {available} are visible."
        )

    def _measure(spec, N, num_gpus):
        """Integrate one config at resolution ``N`` on ``num_gpus`` devices.

        Returns ``(runtime, temporary_bytes, argument_bytes, total_bytes)``. A
        run that fails (typically out of device memory at the largest N) is
        reported and turned into a NaN runtime / zero byte counts so the sweep
        can continue and still plot the resolutions that did fit.
        """
        try:
            r, *_ = _run_simulation(spec, N, setup_fn, num_gpus=num_gpus)
            return (
                float(r.runtime),
                int(r.temporary_memory_bytes),
                int(r.argument_memory_bytes),
                int(r.total_memory_bytes),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] {spec.label} N={N} on {num_gpus} GPU(s) FAILED: {exc!r}")
            return (np.nan, 0, 0, 0)

    # -------------------------------------------------------------
    # ==== ↓ Measure runtime + memory across the resolution sweep ↓ =
    # -------------------------------------------------------------

    # For each benchmark and resolution we time a single-GPU run and a
    # ``num_gpus``-GPU run, keeping both runtimes and the per-device memory
    # figures so the strong-scaling speedup and memory panels can be plotted.
    out = {}
    for spec in benchmarks:
        rec = dict(
            N=list(N_values),
            runtime_1=[], runtime_N=[],
            temp_1=[], temp_N=[],
            arg_1=[], arg_N=[],
            total_1=[], total_N=[],
        )
        for N in N_values:
            runtime_single_gpu, temp_single_gpu, arg_single_gpu, total_single_gpu = _measure(
                spec, N, num_gpus=1
            )
            runtime_multi_gpu, temp_multi_gpu, arg_multi_gpu, total_multi_gpu = _measure(
                spec, N, num_gpus=num_gpus
            )
            rec["runtime_1"].append(runtime_single_gpu)
            rec["runtime_N"].append(runtime_multi_gpu)
            rec["temp_1"].append(temp_single_gpu)
            rec["temp_N"].append(temp_multi_gpu)
            rec["arg_1"].append(arg_single_gpu)
            rec["arg_N"].append(arg_multi_gpu)
            rec["total_1"].append(total_single_gpu)
            rec["total_N"].append(total_multi_gpu)
            if (
                np.isfinite(runtime_single_gpu)
                and np.isfinite(runtime_multi_gpu)
                and runtime_multi_gpu > 0
            ):
                speedup_str = f"{runtime_single_gpu / runtime_multi_gpu:.2f}x"
            else:
                speedup_str = "n/a"
            print(
                f"[{name}] {spec.label} N={N}: "
                f"1 GPU={runtime_single_gpu:.2f}s, "
                f"{num_gpus} GPUs={runtime_multi_gpu:.2f}s, "
                f"speedup={speedup_str}"
            )
        out[spec.label] = rec

    # -------------------------------------------------------------
    # ==== ↑ Measure runtime + memory across the resolution sweep ↑ =
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============= ↓ Build the strong-scaling figure ↓ ===========
    # -------------------------------------------------------------

    MB = 1024 ** 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_t, ax_s = axes[0]
    ax_temp, ax_arg = axes[1]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for idx, (label, rec) in enumerate(out.items()):
        color = color_cycle[idx % len(color_cycle)]
        runtime_single = np.array(rec["runtime_1"])
        runtime_multi = np.array(rec["runtime_N"])
        speedup = runtime_single / runtime_multi

        ax_t.loglog(
            N_values,
            runtime_single,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"{label}, 1 GPU",
        )
        ax_t.loglog(
            N_values,
            runtime_multi,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{label}, {num_gpus} GPUs",
        )

        ax_s.plot(
            N_values,
            speedup,
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )

        ax_temp.loglog(
            N_values,
            np.array(rec["temp_1"]) / MB,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"{label}, 1 GPU",
        )
        ax_temp.loglog(
            N_values,
            np.array(rec["temp_N"]) / MB,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{label}, {num_gpus} GPUs (per device)",
        )

        ax_arg.loglog(
            N_values,
            np.array(rec["arg_1"]) / MB,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"{label}, 1 GPU",
        )
        ax_arg.loglog(
            N_values,
            np.array(rec["arg_N"]) / MB,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{label}, {num_gpus} GPUs (per device)",
        )

    ax_s.axhline(
        num_gpus,
        color="k",
        linestyle="--",
        alpha=0.7,
        label=f"ideal speedup ({num_gpus}x)",
    )

    ax_t.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_t.set_ylabel("Runtime (s)", fontsize=12)
    ax_t.set_title(f"{title}: strong-scaling runtime", fontsize=13)
    ax_s.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_s.set_ylabel(f"Speedup ($T_1 / T_{{{num_gpus}}}$)", fontsize=12)
    ax_s.set_title("Strong-scaling speedup", fontsize=13)
    ax_temp.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_temp.set_ylabel("Temporary memory per device (MB)", fontsize=12)
    ax_temp.set_title("Compiled-step temporary memory", fontsize=13)
    ax_arg.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_arg.set_ylabel("Argument memory per device (MB)", fontsize=12)
    ax_arg.set_title("Compiled-step argument memory", fontsize=13)

    for ax in (ax_t, ax_s, ax_temp, ax_arg):
        _format_xaxis(ax, N_values)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="major", ls="-", alpha=0.2)
    # The speedup panel stays on a linear scale, and its upper limit tracks the
    # largest speedup actually observed (falling back to the ideal factor) so
    # the ideal-speedup guide line always stays in view.
    ax_s.set_yscale("linear")
    speedups_seen = []
    for record in out.values():
        speedup_curve = np.array(record["runtime_1"]) / np.array(record["runtime_N"])
        speedup_curve = speedup_curve[np.isfinite(speedup_curve)]
        if speedup_curve.size:
            speedups_seen.append(float(speedup_curve.max()))
    max_speedup_seen = max(speedups_seen) if speedups_seen else float(num_gpus)
    ax_s.set_ylim(0, max(num_gpus, max_speedup_seen) * 1.1)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{name}_strong_scaling.svg"))

    # -------------------------------------------------------------
    # ============= ↑ Build the strong-scaling figure ↑ ===========
    # -------------------------------------------------------------

    flat = {"N_values": np.array(N_values), "num_gpus": num_gpus}
    for label, rec in out.items():
        slug = _slug(label)
        for field_name, values in rec.items():
            if field_name == "N":
                continue
            flat[f"{slug}__{field_name}"] = np.array(values)
    np.savez(os.path.join(data_dir, f"{name}_strong_scaling.npz"), **flat)

    return out
