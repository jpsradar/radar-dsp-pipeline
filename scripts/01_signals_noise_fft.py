#!/usr/bin/env python3
"""
01_signals_noise_fft.py

Stage 01 of the radar DSP pipeline: signal generation, additive noise, FFT
analysis, spectral peak detection, and detectability estimation.

Pipeline role
-------------
This script demonstrates the first DSP concept required for radar-oriented
processing:

    signal buried in noise → FFT spectral peak → detectability metric

It intentionally acts as a thin executable stage. Signal generation and FFT
logic live in reusable modules under `src/`.

Engineering purpose
-------------------
The goal is not merely to plot an FFT. The goal is to quantify how a known
sinusoidal component becomes visible, or fails to become visible, as SNR changes.

This directly supports later stages:
- filtering and bandwidth trade-offs
- windowing and spectral leakage analysis
- Doppler peak detection
- thresholding and false-alarm behavior
- system-level DSP trade-off analysis

Inputs
------
Configured through CLI arguments:
- sampling frequency
- signal duration
- sinusoid frequency
- SNR values
- random seed
- output directory

Outputs
-------
Generated under `figures/generated_plots/` by default:
- 01_multi_snr_spectrum.png
- 01_detectability_vs_snr.png

Usage
-----
Pipeline mode:
    python scripts/01_signals_noise_fft.py

Custom SNR sweep:
    python scripts/01_signals_noise_fft.py --snr-db -30 -20 -10 0 10

Interactive inspection:
    python scripts/01_signals_noise_fft.py --show
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


DEFAULT_OUTPUT_DIR = Path("figures/generated_plots")


@dataclass(frozen=True)
class SpectralAnalysisResult:
    """
    Container for one SNR point in the spectral detectability sweep.

    Attributes
    ----------
    snr_db:
        Time-domain signal-to-noise ratio used to generate the noisy signal.
    freq_hz:
        Single-sided FFT frequency axis.
    magnitude_db:
        Single-sided magnitude spectrum in decibels.
    peak_frequency_hz:
        Frequency of the detected spectral peak.
    peak_magnitude_db:
        Magnitude of the detected spectral peak in dB.
    noise_floor_db:
        Estimated spectral noise floor in dB.
    detectability_db:
        Difference between detected peak magnitude and estimated noise floor.
    peak_bin_index:
        FFT bin index of the detected spectral peak.
    """

    snr_db: float
    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    peak_frequency_hz: float
    peak_magnitude_db: float
    noise_floor_db: float
    detectability_db: float
    peak_bin_index: int


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Stage 01 pipeline script.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stage 01 radar DSP pipeline: multi-SNR signal generation, FFT, "
            "spectral peak detection, and detectability analysis."
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
        help="Sinusoid frequency in hertz. Default: 50.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Sinusoid peak amplitude. Default: 1.0.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[-20.0, -10.0, 0.0, 10.0],
        help="SNR values in dB for the sweep. Default: -20 -10 0 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for reproducible AWGN generation. Default: 123.",
    )
    parser.add_argument(
        "--search-half-width-hz",
        type=float,
        default=10.0,
        help=(
            "Half-width of the peak search band around the expected signal "
            "frequency. Default: 10 Hz."
        ),
    )
    parser.add_argument(
        "--noise-guard-bins",
        type=int,
        default=5,
        help="FFT bins excluded around the detected peak for noise-floor estimation.",
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


def analyze_single_snr(args: argparse.Namespace, snr_db: float, seed: int) -> SpectralAnalysisResult:
    """
    Generate one noisy signal and compute spectral detectability metrics.

    Parameters
    ----------
    args:
        Parsed pipeline configuration.
    snr_db:
        SNR value used for this sweep point.
    seed:
        Random seed used for this sweep point.

    Returns
    -------
    SpectralAnalysisResult
        FFT spectrum and detectability metrics for the given SNR.
    """
    _, noisy_signal, _, _ = generate_noisy_sinusoid(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        frequency_hz=args.frequency_hz,
        snr_db=snr_db,
        amplitude=args.amplitude,
        seed=seed,
    )

    freq_hz, magnitude = compute_single_sided_fft(
        noisy_signal,
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

    return SpectralAnalysisResult(
        snr_db=snr_db,
        freq_hz=freq_hz,
        magnitude_db=magnitude_db,
        peak_frequency_hz=peak.frequency_hz,
        peak_magnitude_db=peak.magnitude_db,
        noise_floor_db=noise_floor_db,
        detectability_db=detectability_db,
        peak_bin_index=peak.bin_index,
    )


def run_snr_sweep(args: argparse.Namespace) -> list[SpectralAnalysisResult]:
    """
    Run the configured multi-SNR spectral detectability sweep.

    Parameters
    ----------
    args:
        Parsed pipeline configuration.

    Returns
    -------
    list[SpectralAnalysisResult]
        One result object per SNR value.
    """
    results: list[SpectralAnalysisResult] = []

    for idx, snr_db in enumerate(args.snr_db):
        seed = args.seed + idx
        result = analyze_single_snr(args, snr_db=snr_db, seed=seed)
        results.append(result)

        print(
            "SNR={snr:>7.2f} dB | "
            "Peak={peak:>8.2f} dB @ {freq:>8.2f} Hz | "
            "Floor={floor:>8.2f} dB | "
            "Detectability={det:>8.2f} dB".format(
                snr=result.snr_db,
                peak=result.peak_magnitude_db,
                freq=result.peak_frequency_hz,
                floor=result.noise_floor_db,
                det=result.detectability_db,
            )
        )

    return results


def plot_multi_snr_spectrum(
    results: list[SpectralAnalysisResult],
    output_dir: Path,
) -> Path:
    """
    Plot the multi-SNR magnitude spectra and detected spectral peaks.

    Parameters
    ----------
    results:
        Spectral analysis results for all SNR values.
    output_dir:
        Directory where the plot will be written.

    Returns
    -------
    Path
        Path to the generated figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_multi_snr_spectrum.png"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for result in results:
        ax.plot(
            result.freq_hz,
            result.magnitude_db,
            linewidth=1.8,
            label=f"SNR = {result.snr_db:g} dB",
        )
        ax.scatter(
            result.peak_frequency_hz,
            result.peak_magnitude_db,
            marker="x",
            s=70,
        )

    ax.set_title("Multi-SNR Spectrum: Signal Peak Visibility")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_detectability_vs_snr(
    results: list[SpectralAnalysisResult],
    output_dir: Path,
) -> Path:
    """
    Plot spectral detectability as a function of input SNR.

    Parameters
    ----------
    results:
        Spectral analysis results for all SNR values.
    output_dir:
        Directory where the plot will be written.

    Returns
    -------
    Path
        Path to the generated figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_detectability_vs_snr.png"

    snr_values = np.array([r.snr_db for r in results], dtype=float)
    detectability = np.array([r.detectability_db for r in results], dtype=float)

    order = np.argsort(snr_values)
    snr_values = snr_values[order]
    detectability = detectability[order]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    ax.plot(snr_values, detectability, marker="o", linewidth=2.2)

    ax.set_title("Spectral Detectability vs Input SNR")
    ax.set_xlabel("Input SNR [dB]")
    ax.set_ylabel("Peak above noise floor [dB]")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def generate_figures(
    results: list[SpectralAnalysisResult],
    output_dir: Path,
    *,
    show: bool,
) -> list[Path]:
    """
    Generate all Stage 01 figures.

    Parameters
    ----------
    results:
        Spectral analysis results for all SNR values.
    output_dir:
        Directory where generated figures will be written.
    show:
        If True, open generated figures interactively after saving.

    Returns
    -------
    list[Path]
        Paths to generated figure artifacts.
    """
    figure_paths = [
        plot_multi_snr_spectrum(results, output_dir),
        plot_detectability_vs_snr(results, output_dir),
    ]

    if show:
        for path in figure_paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10.0, 6.0))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()

    return figure_paths


def main() -> int:
    """
    Execute Stage 01 of the radar DSP pipeline.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()

    results = run_snr_sweep(args)
    figure_paths = generate_figures(
        results,
        args.output_dir,
        show=args.show,
    )

    print("\n[OK] Generated Stage 01 artifacts:")
    for path in figure_paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())