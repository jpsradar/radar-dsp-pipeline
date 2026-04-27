#!/usr/bin/env python3
"""
02_fir_iir_filters.py

Stage 02 of the radar DSP pipeline: FIR/IIR low-pass filtering and its effect
on spectral detectability.

Pipeline role
-------------
This script demonstrates how filtering changes the spectral environment around
a signal of interest:

    noisy signal → low-pass filtering → spectrum shaping → detectability impact

It intentionally acts as a thin executable stage. Signal generation, FFT,
detectability metrics, and filtering utilities live in reusable modules under
`src/`.

Engineering purpose
-------------------
Filtering is not shown here as a cosmetic smoothing operation. It is treated as
a system design choice that affects:

- noise bandwidth
- out-of-band rejection
- signal visibility in the spectrum
- implementation trade-offs between FIR and IIR filters

Downstream relevance
--------------------
This stage supports later work on:

- spectral leakage and windowing
- Doppler peak isolation
- threshold detection
- system-level DSP trade-offs

Inputs
------
Configured through CLI arguments:

- sampling frequency
- signal duration
- signal frequency
- input SNR
- filter cutoff
- FIR tap count
- IIR order
- random seed
- output directory

Outputs
-------
Generated under `figures/generated_plots/` by default:

- 02_filter_response_fir_vs_iir.png
- 02_filtered_signal_spectrum.png
- 02_filter_detectability_summary.png

Usage
-----
Pipeline mode:
    python scripts/02_fir_iir_filters.py

Interactive inspection:
    python scripts/02_fir_iir_filters.py --show
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
from src.filters import (
    apply_filter,
    compute_frequency_response,
    design_fir_lowpass,
    design_iir_lowpass,
)
from src.signals import generate_noisy_sinusoid


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated_plots"


@dataclass(frozen=True)
class FilteredSignalResult:
    """
    Spectral analysis result for one signal condition.

    Attributes
    ----------
    label:
        Human-readable label identifying the signal condition.
    freq_hz:
        Frequency axis in hertz.
    magnitude_db:
        Single-sided magnitude spectrum in dB.
    peak_frequency_hz:
        Detected spectral peak frequency.
    peak_magnitude_db:
        Detected spectral peak magnitude in dB.
    noise_floor_db:
        Estimated spectral noise floor in dB.
    detectability_db:
        Peak-to-noise-floor metric in dB.
    """

    label: str
    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    peak_frequency_hz: float
    peak_magnitude_db: float
    noise_floor_db: float
    detectability_db: float


@dataclass(frozen=True)
class FilterDesignResult:
    """
    FIR/IIR filter coefficients and frequency responses.

    Attributes
    ----------
    fir_b:
        FIR numerator coefficients.
    iir_b:
        IIR numerator coefficients.
    iir_a:
        IIR denominator coefficients.
    fir_freq_hz:
        FIR frequency response axis.
    fir_response_db:
        FIR magnitude response in dB.
    iir_freq_hz:
        IIR frequency response axis.
    iir_response_db:
        IIR magnitude response in dB.
    """

    fir_b: np.ndarray
    iir_b: np.ndarray
    iir_a: np.ndarray
    fir_freq_hz: np.ndarray
    fir_response_db: np.ndarray
    iir_freq_hz: np.ndarray
    iir_response_db: np.ndarray


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Stage 02.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stage 02 radar DSP pipeline: FIR/IIR filtering, spectral shaping, "
            "and detectability impact analysis."
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
        default=50.0,
        help="Signal frequency in hertz. Default: 50.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=-10.0,
        help="Input time-domain SNR in dB. Default: -10.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=120.0,
        help="Low-pass filter cutoff frequency in hertz. Default: 120.",
    )
    parser.add_argument(
        "--fir-taps",
        type=int,
        default=81,
        help="Number of FIR taps. Default: 81.",
    )
    parser.add_argument(
        "--iir-order",
        type=int,
        default=4,
        help="Butterworth IIR order. Default: 4.",
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
        default=5,
        help="Bins excluded around detected peak for noise-floor estimation. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible AWGN. Default: 123.",
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


def validate_filter_configuration(args: argparse.Namespace) -> None:
    """
    Validate the filter configuration before running the stage.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Raises
    ------
    ValueError
        If the filter does not preserve the target signal frequency or violates
        sampling constraints.
    """
    if args.cutoff_hz <= args.frequency_hz:
        raise ValueError(
            "cutoff_hz must be greater than frequency_hz so the signal of "
            "interest remains inside the passband."
        )

    if args.cutoff_hz >= args.fs_hz / 2.0:
        raise ValueError("cutoff_hz must be below Nyquist frequency.")

    if args.fir_taps < 3:
        raise ValueError("fir_taps must be at least 3.")

    if args.iir_order < 1:
        raise ValueError("iir_order must be at least 1.")


def design_filters(args: argparse.Namespace) -> FilterDesignResult:
    """
    Design FIR and IIR low-pass filters and compute their responses.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Returns
    -------
    FilterDesignResult
        Filter coefficients and magnitude responses.
    """
    fir_b = design_fir_lowpass(
        cutoff_hz=args.cutoff_hz,
        fs_hz=args.fs_hz,
        num_taps=args.fir_taps,
        window="hann",
    )

    iir_b, iir_a = design_iir_lowpass(
        cutoff_hz=args.cutoff_hz,
        fs_hz=args.fs_hz,
        order=args.iir_order,
    )

    fir_freq_hz, fir_response_db = compute_frequency_response(
        fir_b,
        fs_hz=args.fs_hz,
        worN=2048,
    )

    iir_freq_hz, iir_response_db = compute_frequency_response(
        iir_b,
        a=iir_a,
        fs_hz=args.fs_hz,
        worN=2048,
    )

    return FilterDesignResult(
        fir_b=fir_b,
        iir_b=iir_b,
        iir_a=iir_a,
        fir_freq_hz=fir_freq_hz,
        fir_response_db=fir_response_db,
        iir_freq_hz=iir_freq_hz,
        iir_response_db=iir_response_db,
    )


def analyze_signal_condition(
    *,
    label: str,
    signal: np.ndarray,
    args: argparse.Namespace,
) -> FilteredSignalResult:
    """
    Compute spectrum and detectability metrics for one signal condition.

    Parameters
    ----------
    label:
        Signal condition label.
    signal:
        Time-domain signal to analyze.
    args:
        Parsed runtime configuration.

    Returns
    -------
    FilteredSignalResult
        Frequency spectrum and detectability metrics.
    """
    freq_hz, magnitude = compute_single_sided_fft(
        signal,
        fs_hz=args.fs_hz,
        remove_dc=True,
    )

    magnitude_db = magnitude_to_db(magnitude)

    peak = detect_spectral_peak(
        freq_hz,
        magnitude,
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

    return FilteredSignalResult(
        label=label,
        freq_hz=freq_hz,
        magnitude_db=magnitude_db,
        peak_frequency_hz=peak.frequency_hz,
        peak_magnitude_db=peak.magnitude_db,
        noise_floor_db=noise_floor_db,
        detectability_db=detectability_db,
    )


def run_filter_analysis(args: argparse.Namespace) -> tuple[FilterDesignResult, list[FilteredSignalResult]]:
    """
    Run the full Stage 02 filtering analysis.

    Parameters
    ----------
    args:
        Parsed runtime configuration.

    Returns
    -------
    tuple[FilterDesignResult, list[FilteredSignalResult]]
        Designed filters and spectral analysis results.
    """
    validate_filter_configuration(args)

    _, noisy_signal, _, _ = generate_noisy_sinusoid(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        frequency_hz=args.frequency_hz,
        snr_db=args.snr_db,
        seed=args.seed,
    )

    filters = design_filters(args)

    fir_signal = apply_filter(noisy_signal, filters.fir_b)
    iir_signal = apply_filter(noisy_signal, filters.iir_b, filters.iir_a)

    results = [
        analyze_signal_condition(label="Unfiltered", signal=noisy_signal, args=args),
        analyze_signal_condition(label="FIR low-pass", signal=fir_signal, args=args),
        analyze_signal_condition(label="IIR low-pass", signal=iir_signal, args=args),
    ]

    return filters, results


def plot_filter_response(filters: FilterDesignResult, args: argparse.Namespace) -> Path:
    """
    Plot FIR and IIR low-pass magnitude responses.

    Parameters
    ----------
    filters:
        Filter design output.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "02_filter_response_fir_vs_iir.png"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    ax.plot(filters.fir_freq_hz, filters.fir_response_db, linewidth=2.0, label="FIR low-pass")
    ax.plot(filters.iir_freq_hz, filters.iir_response_db, linewidth=2.0, label="IIR low-pass")
    ax.axvline(args.frequency_hz, linestyle="--", linewidth=1.4, label="Signal frequency")
    ax.axvline(args.cutoff_hz, linestyle=":", linewidth=1.6, label="Cutoff frequency")

    ax.set_title("FIR vs IIR Low-Pass Filter Response")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_ylim(-90.0, 5.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_filtered_signal_spectrum(results: list[FilteredSignalResult], args: argparse.Namespace) -> Path:
    """
    Plot spectra before and after FIR/IIR filtering.

    Parameters
    ----------
    results:
        Spectral analysis results for unfiltered and filtered signals.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "02_filtered_signal_spectrum.png"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for result in results:
        ax.plot(
            result.freq_hz,
            result.magnitude_db,
            linewidth=1.8,
            label=(
                f"{result.label} "
                f"(detectability={result.detectability_db:.1f} dB)"
            ),
        )
        ax.scatter(
            result.peak_frequency_hz,
            result.peak_magnitude_db,
            marker="x",
            s=70,
        )

    ax.set_title("Filtered Signal Spectrum: Noise Bandwidth vs Detectability")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_detectability_summary(results: list[FilteredSignalResult], args: argparse.Namespace) -> Path:
    """
    Plot detectability improvement for each filtering condition.

    Parameters
    ----------
    results:
        Spectral analysis results for all signal conditions.
    args:
        Parsed runtime configuration.

    Returns
    -------
    Path
        Path to generated figure.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "02_filter_detectability_summary.png"

    labels = [result.label for result in results]
    detectability = [result.detectability_db for result in results]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    bars = ax.bar(labels, detectability)

    ax.set_title("Detectability Impact of FIR/IIR Filtering")
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


def generate_figures(
    filters: FilterDesignResult,
    results: list[FilteredSignalResult],
    args: argparse.Namespace,
) -> list[Path]:
    """
    Generate all Stage 02 figures.

    Parameters
    ----------
    filters:
        Designed filters and frequency responses.
    results:
        Spectral analysis results.
    args:
        Parsed runtime configuration.

    Returns
    -------
    list[Path]
        Generated figure paths.
    """
    figure_paths = [
        plot_filter_response(filters, args),
        plot_filtered_signal_spectrum(results, args),
        plot_detectability_summary(results, args),
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


def print_summary(results: list[FilteredSignalResult]) -> None:
    """
    Print a compact numerical summary of Stage 02 results.

    Parameters
    ----------
    results:
        Spectral analysis results.
    """
    print("\n[OK] Stage 02 detectability summary")
    for result in results:
        print(
            "{label:<14} | Peak={peak:>8.2f} dB @ {freq:>8.2f} Hz | "
            "Floor={floor:>8.2f} dB | Detectability={det:>8.2f} dB".format(
                label=result.label,
                peak=result.peak_magnitude_db,
                freq=result.peak_frequency_hz,
                floor=result.noise_floor_db,
                det=result.detectability_db,
            )
        )


def main() -> int:
    """
    Execute Stage 02 of the radar DSP pipeline.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()

    filters, results = run_filter_analysis(args)
    print_summary(results)

    figure_paths = generate_figures(filters, results, args)

    print("\n[OK] Generated Stage 02 artifacts:")
    for path in figure_paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())