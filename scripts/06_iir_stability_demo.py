#!/usr/bin/env python3
"""
06_iir_stability_demo.py

Stage 06 of the radar DSP pipeline: practical IIR stability and numerical
robustness demonstration.

Pipeline role
-------------
This script compares recursive-filter behavior for:

- stable poles
- poles near the unit circle
- unstable poles
- direct-form implementation
- second-order-section implementation
- float32 and float64 arithmetic

The numerical primitives live in `src.iir_stability`. This executable stage
handles configuration, orchestration, visualization, and reporting.

Inputs
------
Configured through CLI arguments:

- impulse-response length
- repeated second-order-section count
- conjugate-pole angle
- stable pole radius
- near-unit-circle pole radius
- unstable pole radius
- output directory

Outputs
-------
Generated under `figures/generated_plots/` by default:

- 06_iir_stability_impulse_response.png

Usage
-----
Pipeline mode:
    python scripts/06_iir_stability_demo.py

Interactive inspection:
    python scripts/06_iir_stability_demo.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.iir_stability import (
    StabilityCase,
    StabilityResult,
    run_stability_case,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated_plots"
MIN_SAMPLES = 32
MAX_SAFE_PLOT_MAGNITUDE = 1.0e100
LOG_FLOOR = 1.0e-300


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Stage 06.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stage 06 radar DSP pipeline: IIR pole stability and numerical "
            "robustness demonstration."
        )
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Impulse-response length in samples. Default: 2000.",
    )
    parser.add_argument(
        "--section-count",
        type=int,
        default=4,
        help="Number of repeated second-order sections. Default: 4.",
    )
    parser.add_argument(
        "--pole-angle-deg",
        type=float,
        default=20.0,
        help="Conjugate-pole angle in degrees. Default: 20.",
    )
    parser.add_argument(
        "--stable-radius",
        type=float,
        default=0.90,
        help="Stable pole radius. Default: 0.90.",
    )
    parser.add_argument(
        "--near-unit-radius",
        type=float,
        default=0.999,
        help="Near-unit-circle pole radius. Default: 0.999.",
    )
    parser.add_argument(
        "--unstable-radius",
        type=float,
        default=1.01,
        help="Unstable pole radius. Default: 1.01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively. By default, figures are saved only.",
    )

    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> None:
    """
    Validate the Stage 06 experiment configuration.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Raises
    ------
    ValueError
        If sample count, section count, pole angle, or pole radii are invalid.
    """
    if args.num_samples < MIN_SAMPLES:
        raise ValueError(
            f"num_samples must be at least {MIN_SAMPLES} samples."
        )

    if args.section_count < 1:
        raise ValueError("section_count must be at least 1.")

    if not 0.0 < args.pole_angle_deg < 180.0:
        raise ValueError("pole_angle_deg must be within (0, 180) degrees.")

    if args.stable_radius <= 0.0:
        raise ValueError("stable_radius must be positive.")

    if args.near_unit_radius <= 0.0:
        raise ValueError("near_unit_radius must be positive.")

    if args.unstable_radius <= 0.0:
        raise ValueError("unstable_radius must be positive.")

    if args.stable_radius >= 1.0:
        raise ValueError("stable_radius must be below 1.")

    if not args.stable_radius < args.near_unit_radius < 1.0:
        raise ValueError(
            "near_unit_radius must be greater than stable_radius and below 1."
        )

    if args.unstable_radius <= 1.0:
        raise ValueError("unstable_radius must be greater than 1.")


def run_experiment(args: argparse.Namespace) -> list[StabilityResult]:
    """
    Run the complete IIR stability experiment.

    Parameters
    ----------
    args:
        Validated runtime configuration.

    Returns
    -------
    list[StabilityResult]
        Results for every pole-radius, implementation, and dtype combination.
    """
    validate_configuration(args)

    cases = [
        StabilityCase(
            label="Stable",
            pole_radius=args.stable_radius,
        ),
        StabilityCase(
            label="Near unit circle",
            pole_radius=args.near_unit_radius,
        ),
        StabilityCase(
            label="Unstable",
            pole_radius=args.unstable_radius,
        ),
    ]

    pole_angle_rad = float(np.deg2rad(args.pole_angle_deg))
    dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    results: list[StabilityResult] = []

    for case in cases:
        for dtype in dtypes:
            results.extend(
                run_stability_case(
                    case=case,
                    pole_angle_rad=pole_angle_rad,
                    section_count=args.section_count,
                    num_samples=args.num_samples,
                    dtype=dtype,
                )
            )

    return results


def sanitize_output(output: np.ndarray) -> np.ndarray:
    """
    Replace non-finite and extreme values for safe visualization.

    Parameters
    ----------
    output:
        Time-domain filter output.

    Returns
    -------
    np.ndarray
        Finite float64 array clipped to the plotting safety range.
    """
    output_float64 = np.asarray(output, dtype=np.float64)

    sanitized = np.nan_to_num(
        output_float64,
        nan=0.0,
        posinf=MAX_SAFE_PLOT_MAGNITUDE,
        neginf=-MAX_SAFE_PLOT_MAGNITUDE,
    )

    return np.clip(
        sanitized,
        -MAX_SAFE_PLOT_MAGNITUDE,
        MAX_SAFE_PLOT_MAGNITUDE,
    )


def plot_impulse_responses(
    results: list[StabilityResult],
    args: argparse.Namespace,
) -> Path:
    """
    Plot absolute impulse-response magnitudes on a logarithmic scale.

    Parameters
    ----------
    results:
        Numerical stability results.
    args:
        Runtime configuration.

    Returns
    -------
    Path
        Path to the generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "06_iir_stability_impulse_response.png"

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    sample_index = np.arange(args.num_samples)

    for result in results:
        sanitized = sanitize_output(result.output)
        magnitude = np.maximum(np.abs(sanitized), LOG_FLOOR)

        ax.semilogy(
            sample_index,
            magnitude,
            linewidth=1.2,
            label=(
                f"{result.case_label}, r={result.pole_radius:.3f}, "
                f"{result.implementation}, {result.dtype_name}"
            ),
        )

    ax.set_title("IIR Stability: Impulse-Response Magnitude")
    ax.set_xlabel("Sample index [samples]")
    ax.set_ylabel("Absolute output magnitude [linear]")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def print_summary(results: list[StabilityResult]) -> None:
    """
    Print a compact numerical summary of Stage 06 results.

    Parameters
    ----------
    results:
        Numerical stability results.
    """
    print("\n[OK] Stage 06 IIR stability summary")

    for result in results:
        print(
            "{case:<17} | r={radius:>6.3f} | {impl:<11} | "
            "{dtype:<7} | Peak={peak:>12.4e} | "
            "Final={final:>12.4e} | Finite={finite}".format(
                case=result.case_label,
                radius=result.pole_radius,
                impl=result.implementation,
                dtype=result.dtype_name,
                peak=result.peak_magnitude,
                final=result.final_magnitude,
                finite=result.all_finite,
            )
        )


def main() -> int:
    """
    Execute Stage 06 of the radar DSP pipeline.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()

    results = run_experiment(args)
    print_summary(results)

    figure_path = plot_impulse_responses(results, args)

    try:
        display_path = figure_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = figure_path

    print("\n[OK] Generated Stage 06 artifact:")
    print(f"  {display_path}")

    if args.show:
        image = plt.imread(figure_path)
        fig, ax = plt.subplots(figsize=(11.0, 6.2))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(figure_path.name)
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
