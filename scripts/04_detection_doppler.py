#!/usr/bin/env python3
"""
04_detection_doppler.py

Stage 04 of the radar DSP pipeline: Doppler-like spectral peak detection,
thresholding, and empirical Pd/Pfa estimation.

Pipeline role
-------------
This stage connects the previous DSP blocks into a detector workflow:

    fixed-noise environment
        -> Doppler-like sinusoid scaled by SNR
        -> windowing
        -> FFT
        -> spectral peak statistic
        -> threshold decision
        -> empirical Pd/Pfa estimation

Radar interpretation
--------------------
The sinusoid represents a simplified Doppler component. This is not a full
pulse-Doppler radar model; it is a controlled detector experiment that exposes
the key DSP/statistical mechanism:

    detection = spectral statistic > threshold

Engineering purpose
-------------------
This script demonstrates:

- Doppler-like spectral peak detection
- threshold selection from noise-only Monte Carlo trials
- empirical probability of false alarm (Pfa)
- empirical probability of detection (Pd)
- SNR-dependent detector transition
- separation between detectability visualization and detection decision logic

Important modeling convention
-----------------------------
Noise power is held fixed across H0 and H1 trials. SNR is varied by scaling the
signal amplitude. This keeps the noise-only threshold statistically consistent
with the signal-present trials.

Outputs
-------
Generated under `figures/generated_plots/` by default:

- 04_doppler_peak_detection.png
- 04_pd_pfa_basic.png

Usage
-----
Pipeline mode:
    python scripts/04_detection_doppler.py

Interactive inspection:
    python scripts/04_detection_doppler.py --show
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

from src.detection import estimate_detection_probability, estimate_false_alarm_rate
from src.fft_tools import compute_single_sided_fft, magnitude_to_db
from src.signals import generate_sinusoid
from src.windows import WindowName, apply_window, coherent_gain


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated_plots"


@dataclass(frozen=True)
class DetectionTrialConfig:
    """
    Immutable configuration shared by detector trials.

    Attributes
    ----------
    fs_hz:
        Sampling frequency in hertz.
    duration_s:
        Coherent processing interval duration in seconds.
    doppler_hz:
        Doppler-like sinusoidal frequency in hertz.
    noise_std:
        Standard deviation of AWGN samples. This is fixed across H0 and H1.
    window_name:
        Window applied before FFT.
    search_half_width_hz:
        Half-width of the spectral search band around doppler_hz.
    """

    fs_hz: float
    duration_s: float
    doppler_hz: float
    noise_std: float
    window_name: WindowName
    search_half_width_hz: float


@dataclass(frozen=True)
class DetectionCurvePoint:
    """
    Empirical detector performance at one SNR point.

    Attributes
    ----------
    snr_db:
        Input signal-to-noise ratio in dB.
    pd:
        Empirical probability of detection.
    pfa:
        Empirical probability of false alarm.
    threshold_db:
        Detection threshold selected from noise-only trials.
    mean_signal_peak_db:
        Mean signal-present peak statistic in dB.
    mean_noise_peak_db:
        Mean noise-only peak statistic in dB.
    """

    snr_db: float
    pd: float
    pfa: float
    threshold_db: float
    mean_signal_peak_db: float
    mean_noise_peak_db: float


@dataclass(frozen=True)
class ExampleDetectionResult:
    """
    Single-trial Doppler detection result used for visualization.

    Attributes
    ----------
    freq_hz:
        FFT frequency axis.
    magnitude_db:
        Coherent-gain-corrected magnitude spectrum in dB.
    peak_frequency_hz:
        Detected spectral peak frequency.
    peak_magnitude_db:
        Detected peak magnitude in dB.
    noise_floor_db:
        Median spectral floor estimate in dB for display context.
    threshold_db:
        Monte Carlo threshold in dB.
    detected:
        Boolean threshold decision.
    """

    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    peak_frequency_hz: float
    peak_magnitude_db: float
    noise_floor_db: float
    threshold_db: float
    detected: bool


def parse_args() -> argparse.Namespace:
    """
    Parse Stage 04 command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stage 04 radar DSP pipeline: Doppler-like spectral peak detection "
            "with empirical Pd/Pfa estimation."
        )
    )

    parser.add_argument("--fs-hz", type=float, default=1000.0)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--doppler-hz", type=float, default=50.5)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[-35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0],
        help="SNR sweep in dB. Default shows detector transition.",
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["rectangular", "hann", "hamming", "blackman"],
    )
    parser.add_argument("--search-half-width-hz", type=float, default=10.0)
    parser.add_argument("--example-snr-db", type=float, default=-20.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")

    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> None:
    """
    Validate Stage 04 runtime configuration.

    Raises
    ------
    ValueError
        If configuration is physically or numerically invalid.
    """
    if args.fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if args.duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if not 0 < args.doppler_hz < args.fs_hz / 2:
        raise ValueError("doppler_hz must be inside (0, fs/2).")
    if args.noise_std <= 0:
        raise ValueError("noise_std must be positive.")
    if not 0 < args.target_pfa < 1:
        raise ValueError("target_pfa must be in (0, 1).")
    if args.n_trials < 50:
        raise ValueError("n_trials must be at least 50.")
    if args.search_half_width_hz <= 0:
        raise ValueError("search_half_width_hz must be positive.")


def build_trial_config(args: argparse.Namespace) -> DetectionTrialConfig:
    """
    Build immutable trial configuration from CLI arguments.
    """
    return DetectionTrialConfig(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        doppler_hz=args.doppler_hz,
        noise_std=args.noise_std,
        window_name=args.window,
        search_half_width_hz=args.search_half_width_hz,
    )


def amplitude_for_snr_db(snr_db: float, noise_std: float) -> float:
    """
    Convert desired time-domain SNR into real sinusoid peak amplitude.

    For a real sinusoid with peak amplitude A:

        signal_power = A² / 2

    For AWGN:

        noise_power = noise_std²

    Therefore:

        A = sqrt(2 * noise_power * SNR_linear)
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt(2.0 * noise_std**2 * snr_linear))


def generate_noise_only_signal(
    *,
    config: DetectionTrialConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one H0/noise-only trial.
    """
    n_samples = int(round(config.fs_hz * config.duration_s))
    if n_samples < 2:
        raise ValueError("Trial must contain at least two samples.")

    return rng.normal(loc=0.0, scale=config.noise_std, size=n_samples)


def generate_signal_present_signal(
    *,
    config: DetectionTrialConfig,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one H1/signal-present trial at fixed noise power.

    SNR is controlled by scaling the sinusoid amplitude, not by changing the
    noise environment. This keeps H0 and H1 statistically consistent.
    """
    amplitude = amplitude_for_snr_db(snr_db, config.noise_std)

    _, clean = generate_sinusoid(
        fs_hz=config.fs_hz,
        duration_s=config.duration_s,
        frequency_hz=config.doppler_hz,
        amplitude=amplitude,
    )

    noise = rng.normal(loc=0.0, scale=config.noise_std, size=clean.shape)
    return clean + noise


def compute_peak_statistic_db(
    signal: np.ndarray,
    config: DetectionTrialConfig,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute the Doppler-band peak statistic for one time-domain signal.

    Returns
    -------
    tuple
        peak_db, peak_frequency_hz, noise_floor_db, freq_hz, magnitude_db
    """
    windowed, window = apply_window(signal, config.window_name)
    cg = coherent_gain(window)

    freq_hz, magnitude = compute_single_sided_fft(
        windowed,
        fs_hz=config.fs_hz,
        remove_dc=True,
    )

    corrected_magnitude = magnitude / cg
    magnitude_db = magnitude_to_db(corrected_magnitude)

    search_min = config.doppler_hz - config.search_half_width_hz
    search_max = config.doppler_hz + config.search_half_width_hz
    mask = (freq_hz >= search_min) & (freq_hz <= search_max)

    if not np.any(mask):
        raise ValueError("Doppler search band contains no FFT bins.")

    candidate_indices = np.flatnonzero(mask)
    peak_index = int(candidate_indices[np.argmax(magnitude_db[candidate_indices])])

    peak_db = float(magnitude_db[peak_index])
    peak_frequency_hz = float(freq_hz[peak_index])

    floor_mask = np.ones_like(magnitude_db, dtype=bool)
    floor_mask[max(0, peak_index - 5): min(magnitude_db.size, peak_index + 6)] = False
    noise_floor_db = float(np.median(magnitude_db[floor_mask]))

    return peak_db, peak_frequency_hz, noise_floor_db, freq_hz, magnitude_db


def estimate_threshold_from_noise(
    *,
    config: DetectionTrialConfig,
    target_pfa: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    Estimate detection threshold from H0 Monte Carlo peak statistics.
    """
    noise_peaks_db = np.empty(n_trials, dtype=float)

    for idx in range(n_trials):
        noise_signal = generate_noise_only_signal(config=config, rng=rng)
        peak_db, _, _, _, _ = compute_peak_statistic_db(noise_signal, config)
        noise_peaks_db[idx] = peak_db

    threshold_db = float(np.quantile(noise_peaks_db, 1.0 - target_pfa))
    return threshold_db, noise_peaks_db


def evaluate_detector_at_snr(
    *,
    config: DetectionTrialConfig,
    snr_db: float,
    threshold_db: float,
    noise_peaks_db: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
) -> DetectionCurvePoint:
    """
    Estimate empirical Pd and Pfa at one SNR point.
    """
    signal_peaks_db = np.empty(n_trials, dtype=float)

    for idx in range(n_trials):
        signal = generate_signal_present_signal(
            config=config,
            snr_db=snr_db,
            rng=rng,
        )
        peak_db, _, _, _, _ = compute_peak_statistic_db(signal, config)
        signal_peaks_db[idx] = peak_db

    pd = estimate_detection_probability(signal_peaks_db, threshold=threshold_db)
    pfa = estimate_false_alarm_rate(noise_peaks_db, threshold=threshold_db)

    return DetectionCurvePoint(
        snr_db=snr_db,
        pd=pd,
        pfa=pfa,
        threshold_db=threshold_db,
        mean_signal_peak_db=float(np.mean(signal_peaks_db)),
        mean_noise_peak_db=float(np.mean(noise_peaks_db)),
    )


def run_pd_pfa_sweep(args: argparse.Namespace) -> list[DetectionCurvePoint]:
    """
    Run empirical Pd/Pfa estimation over the configured SNR sweep.
    """
    validate_configuration(args)

    config = build_trial_config(args)
    rng = np.random.default_rng(args.seed)

    threshold_db, noise_peaks_db = estimate_threshold_from_noise(
        config=config,
        target_pfa=args.target_pfa,
        n_trials=args.n_trials,
        rng=rng,
    )

    return [
        evaluate_detector_at_snr(
            config=config,
            snr_db=snr,
            threshold_db=threshold_db,
            noise_peaks_db=noise_peaks_db,
            n_trials=args.n_trials,
            rng=rng,
        )
        for snr in args.snr_db
    ]


def build_example_detection(args: argparse.Namespace, threshold_db: float) -> ExampleDetectionResult:
    """
    Generate one example spectrum and detector decision for visualization.
    """
    config = build_trial_config(args)
    rng = np.random.default_rng(args.seed + 10_000)

    signal = generate_signal_present_signal(
        config=config,
        snr_db=args.example_snr_db,
        rng=rng,
    )

    peak_db, peak_freq, noise_floor_db, freq_hz, magnitude_db = compute_peak_statistic_db(
        signal,
        config,
    )

    return ExampleDetectionResult(
        freq_hz=freq_hz,
        magnitude_db=magnitude_db,
        peak_frequency_hz=peak_freq,
        peak_magnitude_db=peak_db,
        noise_floor_db=noise_floor_db,
        threshold_db=threshold_db,
        detected=bool(peak_db > threshold_db),
    )


def plot_doppler_peak_detection(example: ExampleDetectionResult, args: argparse.Namespace) -> Path:
    """
    Plot one Doppler-like spectral peak and its Monte Carlo threshold decision.

    Noise floor is reported in the annotation instead of drawn as a full line,
    because it can overlap the threshold in a single realization and reduce
    readability.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "04_doppler_peak_detection.png"

    x_min = max(0.0, args.doppler_hz - 35.0)
    x_max = args.doppler_hz + 35.0
    band_mask = (example.freq_hz >= x_min) & (example.freq_hz <= x_max)

    if not np.any(band_mask):
        raise ValueError("No FFT bins inside display band.")

    y_band = example.magnitude_db[band_mask]
    y_top = max(example.peak_magnitude_db, example.threshold_db) + 6.0
    y_bottom = max(float(np.nanmin(y_band)) - 5.0, y_top - 60.0)

    status = "DETECTED" if example.detected else "NOT DETECTED"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    ax.plot(example.freq_hz, example.magnitude_db, linewidth=2.0, label="Spectrum")
    ax.axhline(
        example.threshold_db,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="MC threshold",
    )
    ax.scatter(
        example.peak_frequency_hz,
        example.peak_magnitude_db,
        marker="x",
        s=120,
        linewidths=2.4,
        label="Selected peak",
        zorder=5,
    )

    margin_db = example.peak_magnitude_db - example.threshold_db
    ax.annotate(
        (
            f"{example.peak_frequency_hz:.1f} Hz\n"
            f"peak {example.peak_magnitude_db:.1f} dB\n"
            f"floor {example.noise_floor_db:.1f} dB\n"
            f"margin {margin_db:+.1f} dB"
        ),
        xy=(example.peak_frequency_hz, example.peak_magnitude_db),
        xytext=(14, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.95,
        },
    )

    ax.set_title(f"Doppler Peak Detection — {status} | SNR = {args.example_snr_db:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, y_top)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_pd_pfa_basic(points: list[DetectionCurvePoint], args: argparse.Namespace) -> Path:
    """
    Plot empirical Pd and Pfa against SNR using separate panels.

    Separate panels avoid hiding the Pfa curve under the target-Pfa reference
    line and make the detector transition easier to inspect.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "04_pd_pfa_basic.png"

    snr_db = np.array([p.snr_db for p in points], dtype=float)
    pd = np.array([p.pd for p in points], dtype=float)
    pfa = np.array([p.pfa for p in points], dtype=float)

    order = np.argsort(snr_db)
    snr_db = snr_db[order]
    pd = pd[order]
    pfa = pfa[order]

    fig, (ax_pd, ax_pfa) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10.0, 7.0),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax_pd.plot(snr_db, pd, marker="o", linewidth=2.2)
    ax_pd.set_title("Empirical Probability of Detection vs SNR")
    ax_pd.set_ylabel("Pd")
    ax_pd.set_ylim(-0.03, 1.03)
    ax_pd.grid(True, linestyle="--", alpha=0.35)

    ax_pfa.plot(snr_db, pfa, marker="s", linewidth=2.0, label="Empirical Pfa")
    ax_pfa.axhline(args.target_pfa, linestyle="--", linewidth=1.6, label="Target Pfa")
    ax_pfa.set_title("False-Alarm Control")
    ax_pfa.set_xlabel("Input SNR [dB]")
    ax_pfa.set_ylabel("Pfa")
    ax_pfa.set_ylim(0.0, max(0.12, args.target_pfa * 2.2))
    ax_pfa.grid(True, linestyle="--", alpha=0.35)
    ax_pfa.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def generate_figures(points: list[DetectionCurvePoint], args: argparse.Namespace) -> list[Path]:
    """
    Generate all Stage 04 figures.
    """
    threshold_db = points[0].threshold_db
    example = build_example_detection(args, threshold_db)

    paths = [
        plot_doppler_peak_detection(example, args),
        plot_pd_pfa_basic(points, args),
    ]

    if args.show:
        for path in paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10.0, 6.0))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()

    return paths


def print_summary(points: list[DetectionCurvePoint]) -> None:
    """
    Print compact detector-performance summary.
    """
    print("\n[OK] Stage 04 empirical detection summary")
    for point in points:
        print(
            "SNR={snr:>7.2f} dB | Pd={pd:>6.3f} | Pfa={pfa:>6.3f} | "
            "Threshold={thr:>8.2f} dB | Mean H1 peak={h1:>8.2f} dB | "
            "Mean H0 peak={h0:>8.2f} dB".format(
                snr=point.snr_db,
                pd=point.pd,
                pfa=point.pfa,
                thr=point.threshold_db,
                h1=point.mean_signal_peak_db,
                h0=point.mean_noise_peak_db,
            )
        )


def print_generated_artifacts(paths: list[Path]) -> None:
    """
    Print generated artifacts using repo-relative paths.
    """
    print("\n[OK] Generated Stage 04 artifacts:")
    for path in paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")


def main() -> int:
    """
    Execute Stage 04.
    """
    args = parse_args()

    points = run_pd_pfa_sweep(args)
    print_summary(points)

    paths = generate_figures(points, args)
    print_generated_artifacts(paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())