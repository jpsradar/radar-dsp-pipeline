#!/usr/bin/env python3
"""
03_windowing_leakage.py

Stage 03 of the radar DSP pipeline: spectral leakage, windowing, coherent gain,
and detectability trade-offs.

Pipeline role
-------------
This script demonstrates a core radar DSP issue:

    finite observation time → spectral leakage → window trade-offs

It connects directly to Doppler processing, where a target return may not fall
exactly on an FFT bin. In that case, a rectangular window can spread energy
across adjacent bins and distort detectability. Alternative windows reduce
sidelobes but also change coherent gain and equivalent noise bandwidth (ENBW).

Engineering purpose
-------------------
This stage does not treat windowing as cosmetic plot smoothing. It quantifies
how window choice affects:

- spectral leakage around a target-like sinusoidal component
- apparent peak magnitude
- coherent gain correction
- noise floor behavior
- peak-to-noise-floor detectability

Inputs
------
Configured through CLI arguments:

- sampling frequency
- signal duration
- signal frequency
- input SNR
- FFT-bin offset
- random seed
- output directory

Outputs
-------
Generated under `figures/generated_plots/` by default:

- 03_window_leakage_comparison.png
- 03_detectability_window_tradeoff.png
- 03_window_metrics_summary.png

Usage
-----
Pipeline mode:
    python scripts/03_windowing_leakage.py

Interactive inspection:
    python scripts/03_windowing_leakage.py --show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.fft_tools import (
    compute_detectability_db,
    compute_single_sided_fft,
    detect_spectral_peak,
    estimate_noise_floor,
    magnitude_to_db,
)
from src.signals import generate_noisy_sinusoid
from src.windows import (
    WindowName,
    apply_window,
    coherent_gain,
    equivalent_noise_bandwidth,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated_plots"
DEFAULT_WINDOWS: tuple[WindowName, ...] = ("rectangular", "hann", "hamming", "blackman")


@dataclass(frozen=True)
class WindowAnalysisResult:
    """
    Spectral and window metrics for one windowing condition.

    Attributes
    ----------
    window_name:
        Window type used for the analysis.
    freq_hz:
        Single-sided FFT frequency axis.
    magnitude_db:
        Coherent-gain-corrected magnitude spectrum in decibels.
    peak_frequency_hz:
        Detected spectral peak frequency.
    peak_magnitude_db:
        Detected peak magnitude in dB after coherent-gain correction.
    noise_floor_db:
        Estimated spectral noise floor in dB.
    detectability_db:
        Peak-to-noise-floor metric in dB.
    coherent_gain:
        Mean window gain applied to coherent sinusoidal components.
    enbw_bins:
        Equivalent noise bandwidth in FFT bins.
    """

    window_name: str
    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    peak_frequency_hz: float
    peak_magnitude_db: float
    noise_floor_db: float
    detectability_db: float
    coherent_gain: float
    enbw_bins: float


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Stage 03.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stage 03 radar DSP pipeline: windowing, spectral leakage, "
            "coherent gain, ENBW, and detectability trade-offs."
        )
    )

    parser.add_argument(
        "--fs-hz",
        type=float,
        default=1000.0,
        help="Sampling frequency in hertz. Default: 1000.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1.0,
        help="Signal duration in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=50.5,
        help=(
            "Signal frequency in hertz. Default: 50.5. "
            "A non-bin-centered value intentionally exposes spectral leakage."
        ),
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=-5.0,
        help="Input time-domain SNR in dB. Default: -5.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Sinusoid peak amplitude. Default: 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible AWGN. Default: 123.",
    )
    parser.add_argument(
        "--search-half-width-hz",
        type=float,
        default=10.0,
        help="Half-width of peak search band around expected frequency. Default: 10.",
    )
    parser.add_argument(
        "--noise-guard-bins",
        type=int,
        default=8,
        help=(
            "FFT bins excluded around detected peak when estimating noise floor. "
            "Default: 8."
        ),
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
        help="Display plots interactively. By default, figures are saved only.",
    )

    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> None:
    """
    Validate Stage 03 runtime configuration.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Raises
    ------
    ValueError
        If sampling, duration, signal frequency, or guard-bin configuration is invalid.
    """
    if args.fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive.")
    if args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if args.frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if args.frequency_hz >= args.fs_hz / 2.0:
        raise ValueError("frequency_hz must be below Nyquist frequency.")
    if args.search_half_width_hz <= 0.0:
        raise ValueError("search_half_width_hz must be positive.")
    if args.noise_guard_bins < 0:
        raise ValueError("noise_guard_bins must be non-negative.")


def analyze_window_condition(
    *,
    noisy_signal: np.ndarray,
    window_name: WindowName,
    args: argparse.Namespace,
) -> WindowAnalysisResult:
    """
    Apply one window and compute spectral leakage/detectability metrics.

    Parameters
    ----------
    noisy_signal:
        Input noisy time-domain signal.
    window_name:
        Window type to apply.
    args:
        Parsed runtime configuration.

    Returns
    -------
    WindowAnalysisResult
        Spectral and window metrics for the selected window.

    Notes
    -----
    The magnitude spectrum is corrected by coherent gain. This makes peak
    amplitudes comparable across windows and avoids falsely interpreting window
    attenuation as signal loss.
    """
    windowed_signal, window = apply_window(noisy_signal, window_name)

    cg = coherent_gain(window)
    enbw = equivalent_noise_bandwidth(window)

    freq_hz, magnitude = compute_single_sided_fft(
        windowed_signal,
        fs_hz=args.fs_hz,
        remove_dc=True,
    )

    # Correct coherent sinusoidal amplitude loss before comparing peak levels.
    corrected_magnitude = magnitude / cg
    magnitude_db = magnitude_to_db(corrected_magnitude)

    peak = detect_spectral_peak(
        freq_hz,
        corrected_magnitude,
        search_min_hz=args.frequency_hz - args.search_half_width_hz,
        search_max_hz=args.frequency_hz + args.search_half_width_hz,
    )

    noise_floor_db = estimate_noise_floor(
        magnitude_db,
        exclude_index=peak.bin_index,
        guard_bins=args.noise_guard_bins,
        method="median",
    )

    detectability_db = compute_detectability_db(
        peak_magnitude_db=peak.magnitude_db,
        noise_floor_db=noise_floor_db,
    )

    return WindowAnalysisResult(
        window_name=window_name,
        freq_hz=freq_hz,
        magnitude_db=magnitude_db,
        peak_frequency_hz=peak.frequency_hz,
        peak_magnitude_db=peak.magnitude_db,
        noise_floor_db=noise_floor_db,
        detectability_db=detectability_db,
        coherent_gain=cg,
        enbw_bins=enbw,
    )


def run_windowing_analysis(args: argparse.Namespace) -> list[WindowAnalysisResult]:
    """
    Run the Stage 03 windowing and leakage analysis.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Returns
    -------
    list[WindowAnalysisResult]
        One analysis result per configured window.
    """
    validate_configuration(args)

    _, noisy_signal, _, _ = generate_noisy_sinusoid(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        frequency_hz=args.frequency_hz,
        snr_db=args.snr_db,
        amplitude=args.amplitude,
        seed=args.seed,
    )

    return [
        analyze_window_condition(
            noisy_signal=noisy_signal,
            window_name=window_name,
            args=args,
        )
        for window_name in DEFAULT_WINDOWS
    ]


def plot_window_leakage_comparison(
    results: list[WindowAnalysisResult],
    args: argparse.Namespace,
) -> Path:
    """
    Plot spectra for all windows to compare leakage and peak visibility.

    Parameters
    ----------
    results:
        Window analysis results.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "03_window_leakage_comparison.png"

    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    for result in results:
        ax.plot(
            result.freq_hz,
            result.magnitude_db,
            linewidth=1.8,
            label=(
                f"{result.window_name} "
                f"(ENBW={result.enbw_bins:.2f} bins)"
            ),
        )
        ax.scatter(
            result.peak_frequency_hz,
            result.peak_magnitude_db,
            marker="x",
            s=70,
        )

    ax.axvline(
        args.frequency_hz,
        linestyle="--",
        linewidth=1.4,
        label="True signal frequency",
    )

    ax.set_title("Windowing and Spectral Leakage: Corrected Magnitude Spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlim(max(0.0, args.frequency_hz - 60.0), args.frequency_hz + 60.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_detectability_window_tradeoff(
    results: list[WindowAnalysisResult],
    args: argparse.Namespace,
) -> Path:
    """
    Plot detectability across window types.

    Parameters
    ----------
    results:
        Window analysis results.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "03_detectability_window_tradeoff.png"

    labels = [r.window_name for r in results]
    detectability = [r.detectability_db for r in results]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    bars = ax.bar(labels, detectability)

    ax.set_title("Window Choice vs Spectral Detectability")
    ax.set_xlabel("Window")
    ax.set_ylabel("Peak above noise floor [dB]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, detectability):
        ax.annotate(
            f"{value:.1f} dB",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_window_metrics_summary(
    results: list[WindowAnalysisResult],
    args: argparse.Namespace,
) -> Path:
    """
    Plot coherent gain and ENBW to expose the window trade-off explicitly.

    Parameters
    ----------
    results:
        Window analysis results.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "03_window_metrics_summary.png"

    labels = [r.window_name for r in results]
    coherent_gains = [r.coherent_gain for r in results]
    enbw_values = [r.enbw_bins for r in results]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    bars_cg = ax.bar(x - width / 2.0, coherent_gains, width, label="Coherent gain")
    bars_enbw = ax.bar(x + width / 2.0, enbw_values, width, label="ENBW [bins]")

    ax.set_title("Window Metrics: Coherent Gain vs Equivalent Noise Bandwidth")
    ax.set_xlabel("Window")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    for bars in (bars_cg, bars_enbw):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def generate_figures(
    results: list[WindowAnalysisResult],
    args: argparse.Namespace,
) -> list[Path]:
    """
    Generate all Stage 03 figures.

    Parameters
    ----------
    results:
        Window analysis results.
    args:
        Parsed runtime configuration.

    Returns
    -------
    list[Path]
        Generated figure paths.
    """
    figure_paths = [
        plot_window_leakage_comparison(results, args),
        plot_detectability_window_tradeoff(results, args),
        plot_window_metrics_summary(results, args),
    ]

    if args.show:
        for path in figure_paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10.0, 6.0))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()

    return figure_paths


def print_summary(results: list[WindowAnalysisResult]) -> None:
    """
    Print a compact numerical summary of Stage 03 results.

    Parameters
    ----------
    results:
        Window analysis results.
    """
    print("\n[OK] Stage 03 windowing/leakage summary")
    for result in results:
        print(
            "{name:<12} | Peak={peak:>8.2f} dB @ {freq:>8.2f} Hz | "
            "Floor={floor:>8.2f} dB | Detectability={det:>8.2f} dB | "
            "CG={cg:>5.3f} | ENBW={enbw:>5.2f}".format(
                name=result.window_name,
                peak=result.peak_magnitude_db,
                freq=result.peak_frequency_hz,
                floor=result.noise_floor_db,
                det=result.detectability_db,
                cg=result.coherent_gain,
                enbw=result.enbw_bins,
            )
        )


def print_generated_artifacts(paths: list[Path]) -> None:
    """
    Print generated artifacts using repo-relative paths.

    Parameters
    ----------
    paths:
        Generated file paths.
    """
    print("\n[OK] Generated Stage 03 artifacts:")
    for path in paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")


def main() -> int:
    """
    Execute Stage 03 of the radar DSP pipeline.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()

    results = run_windowing_analysis(args)
    print_summary(results)

    figure_paths = generate_figures(results, args)
    print_generated_artifacts(figure_paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())