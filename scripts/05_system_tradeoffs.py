#!/usr/bin/env python3
"""
05_system_tradeoffs.py

Stage 05 of the radar DSP pipeline: system-level DSP trade-off analysis.

Pipeline role
-------------
This stage integrates the previous reusable DSP modules into a controlled
engineering comparison:

    signal + noise
        -> optional filtering
        -> windowing
        -> FFT
        -> detector statistic at the expected Doppler bin
        -> fixed-threshold detection
        -> CA-CFAR detection
        -> Pd/Pfa/SNR/cost summary

The purpose is to evaluate how practical DSP choices affect detector behavior
under controlled assumptions.

Engineering notes
-----------------
Two detectors are compared:

1. Fixed-threshold detector
   The threshold is estimated from Monte Carlo H0 trials so the empirical Pfa
   is controlled directly.

2. CA-CFAR detector
   The threshold is computed per trial from reference cells around the cell
   under test using power-domain cell averaging. The implementation operates in
   linear power, not dB.

Important convention
--------------------
This stage evaluates the detector at the expected Doppler bin rather than using
the maximum bin inside a search band. Selecting the maximum bin changes the
false-alarm problem into a multiple-comparison problem and invalidates the
standard per-cell CA-CFAR Pfa interpretation.

Outputs
-------
Generated under `figures/generated_plots/` by default:

- 05_fixed_threshold_pd_vs_snr.png
- 05_ca_cfar_pd_vs_snr.png
- 05_false_alarm_control.png
- 05_snr_requirement_summary.png
- 05_pipeline_metrics.csv

Usage
-----
Pipeline mode:
    python scripts/05_system_tradeoffs.py

Interactive inspection:
    python scripts/05_system_tradeoffs.py --show
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.detection import estimate_detection_probability, estimate_false_alarm_rate
from src.fft_tools import compute_single_sided_fft, magnitude_to_db
from src.filters import apply_filter, design_fir_lowpass, design_iir_lowpass
from src.signals import generate_sinusoid
from src.windows import WindowName, apply_window, coherent_gain


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated_plots"

FilterType = Literal["none", "fir_lowpass", "iir_lowpass"]
DetectorType = Literal["fixed_threshold", "ca_cfar"]


@dataclass(frozen=True)
class ProcessingChain:
    """
    DSP processing chain evaluated as a system configuration.

    Attributes
    ----------
    name:
        Human-readable name used in plots and CSV output.
    filter_type:
        Filter family applied before windowing.
    window_name:
        Window applied before FFT.
    cost_units:
        Relative implementation-cost proxy. This is a deterministic comparison
        marker, not measured runtime.
    """

    name: str
    filter_type: FilterType
    window_name: WindowName
    cost_units: float


@dataclass(frozen=True)
class ChainDesign:
    """
    Runtime realization of a processing chain.

    Attributes
    ----------
    chain:
        Processing-chain definition.
    fir_b:
        FIR coefficients when filter_type is "fir_lowpass".
    iir_b:
        IIR numerator coefficients when filter_type is "iir_lowpass".
    iir_a:
        IIR denominator coefficients when filter_type is "iir_lowpass".
    """

    chain: ProcessingChain
    fir_b: np.ndarray | None = None
    iir_b: np.ndarray | None = None
    iir_a: np.ndarray | None = None


@dataclass(frozen=True)
class TrialConfig:
    """
    Monte Carlo trial configuration.

    Attributes
    ----------
    fs_hz:
        Sampling frequency in hertz.
    duration_s:
        Coherent processing interval duration in seconds.
    signal_hz:
        Doppler-like signal frequency in hertz.
    noise_std:
        Fixed AWGN standard deviation used for both H0 and H1.
    """

    fs_hz: float
    duration_s: float
    signal_hz: float
    noise_std: float


@dataclass(frozen=True)
class SpectrumAtCut:
    """
    FFT result at the detector cell under test.

    Attributes
    ----------
    freq_hz:
        Single-sided FFT frequency axis.
    magnitude_db:
        Coherent-gain-corrected magnitude spectrum in dB.
    power:
        Coherent-gain-corrected linear power spectrum.
    cut_index:
        Index of the detector cell under test.
    cut_frequency_hz:
        Frequency of the detector cell under test.
    cut_magnitude_db:
        Magnitude of the detector cell in dB.
    cut_power:
        Linear power of the detector cell.
    """

    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    power: np.ndarray
    cut_index: int
    cut_frequency_hz: float
    cut_magnitude_db: float
    cut_power: float


@dataclass(frozen=True)
class DetectorMetrics:
    """
    Performance metrics for one chain/detector combination.

    Attributes
    ----------
    chain_name:
        Name of the processing chain.
    detector:
        Detector type.
    filter_type:
        Filter family.
    window_name:
        Window name.
    cost_units:
        Relative implementation-cost proxy.
    empirical_pfa:
        Empirical false-alarm probability.
    threshold_db:
        Fixed detector threshold in dB, or mean CA-CFAR threshold in dB.
    mean_noise_stat_db:
        Mean H0 CUT statistic in dB.
    snr_db:
        SNR sweep values.
    pd:
        Empirical probability of detection for each SNR.
    snr_for_target_pd_db:
        First SNR reaching target Pd; None if not reached.
    max_pd:
        Maximum Pd over the configured sweep.
    """

    chain_name: str
    detector: DetectorType
    filter_type: str
    window_name: str
    cost_units: float
    empirical_pfa: float
    threshold_db: float
    mean_noise_stat_db: float
    snr_db: np.ndarray
    pd: np.ndarray
    snr_for_target_pd_db: float | None
    max_pd: float


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    The default SNR grid is intentionally dense around the detector transition
    region. Stage 05 is a pipeline stage, not an exploratory notebook; therefore
    the default configuration must produce informative engineering output without
    requiring users to guess runtime parameters.
    """
    parser = argparse.ArgumentParser(
        description="Stage 05 radar DSP pipeline: system-level DSP trade-off analysis."
    )

    parser.add_argument("--fs-hz", type=float, default=1000.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.25,
        help="CPI duration in seconds. Default avoids trivial Pd saturation.",
    )
    parser.add_argument("--signal-hz", type=float, default=50.5)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[
            -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0,
            -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0,
        ],
        help="SNR sweep in dB. Default is dense around the Pd transition region.",
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--target-pd", type=float, default=0.90)
    parser.add_argument("--n-trials", type=int, default=1200)
    parser.add_argument("--filter-cutoff-hz", type=float, default=120.0)
    parser.add_argument("--fir-taps", type=int, default=81)
    parser.add_argument("--iir-order", type=int, default=4)
    parser.add_argument("--cfar-guard-cells", type=int, default=2)
    parser.add_argument("--cfar-reference-cells", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")

    return parser.parse_args()
    

def validate_configuration(args: argparse.Namespace) -> None:
    """
    Validate runtime configuration before any simulation runs.

    Raises
    ------
    ValueError
        If any argument is outside the supported range.
    """
    if args.fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive.")
    if args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if not 0.0 < args.signal_hz < args.fs_hz / 2.0:
        raise ValueError("signal_hz must be inside (0, fs/2).")
    if args.noise_std <= 0.0:
        raise ValueError("noise_std must be positive.")
    if not 0.0 < args.target_pfa < 1.0:
        raise ValueError("target_pfa must be in (0, 1).")
    if not 0.0 < args.target_pd <= 1.0:
        raise ValueError("target_pd must be in (0, 1].")
    if args.n_trials < 100:
        raise ValueError("n_trials must be at least 100.")
    if not args.signal_hz < args.filter_cutoff_hz < args.fs_hz / 2.0:
        raise ValueError("filter_cutoff_hz must preserve the signal and stay below Nyquist.")
    if args.fir_taps < 3:
        raise ValueError("fir_taps must be at least 3.")
    if args.iir_order < 1:
        raise ValueError("iir_order must be at least 1.")
    if args.cfar_guard_cells < 0:
        raise ValueError("cfar_guard_cells must be non-negative.")
    if args.cfar_reference_cells < 2:
        raise ValueError("cfar_reference_cells must be at least 2.")


def default_processing_chains() -> list[ProcessingChain]:
    """
    Return the processing chains used for the trade-off study.

    Returns
    -------
    list[ProcessingChain]
        Ordered chain definitions.
    """
    return [
        ProcessingChain("No filter + Rectangular", "none", "rectangular", 1.0),
        ProcessingChain("No filter + Hann", "none", "hann", 1.2),
        ProcessingChain("FIR low-pass + Hann", "fir_lowpass", "hann", 4.0),
        ProcessingChain("IIR low-pass + Hann", "iir_lowpass", "hann", 2.0),
        ProcessingChain("FIR low-pass + Blackman", "fir_lowpass", "blackman", 4.3),
    ]


def build_trial_config(args: argparse.Namespace) -> TrialConfig:
    """
    Build immutable trial configuration from CLI arguments.
    """
    return TrialConfig(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        signal_hz=args.signal_hz,
        noise_std=args.noise_std,
    )


def design_processing_chain(chain: ProcessingChain, args: argparse.Namespace) -> ChainDesign:
    """
    Design filters required by one processing chain.

    Parameters
    ----------
    chain:
        Chain definition.
    args:
        Runtime configuration.

    Returns
    -------
    ChainDesign
        Runtime chain design.
    """
    if chain.filter_type == "none":
        return ChainDesign(chain=chain)

    if chain.filter_type == "fir_lowpass":
        fir_b = design_fir_lowpass(
            cutoff_hz=args.filter_cutoff_hz,
            fs_hz=args.fs_hz,
            num_taps=args.fir_taps,
            window="hann",
        )
        return ChainDesign(chain=chain, fir_b=fir_b)

    if chain.filter_type == "iir_lowpass":
        iir_b, iir_a = design_iir_lowpass(
            cutoff_hz=args.filter_cutoff_hz,
            fs_hz=args.fs_hz,
            order=args.iir_order,
        )
        return ChainDesign(chain=chain, iir_b=iir_b, iir_a=iir_a)

    raise ValueError(f"Unsupported filter type: {chain.filter_type!r}")


def amplitude_for_snr_db(snr_db: float, noise_std: float) -> float:
    """
    Convert target time-domain SNR into real sinusoid peak amplitude.

    For a real sinusoid:

        signal_power = A² / 2

    For AWGN:

        noise_power = noise_std²

    Therefore:

        A = sqrt(2 * noise_power * SNR_linear)
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt(2.0 * noise_std**2 * snr_linear))


def generate_noise_only_signal(config: TrialConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one H0/noise-only trial.
    """
    n_samples = int(round(config.fs_hz * config.duration_s))
    if n_samples < 2:
        raise ValueError("Trial must contain at least two samples.")

    return rng.normal(loc=0.0, scale=config.noise_std, size=n_samples)


def generate_signal_present_signal(
    config: TrialConfig,
    *,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one H1/signal-present trial with fixed noise power.

    SNR is varied by scaling the signal amplitude. Noise statistics remain fixed
    across H0 and H1.
    """
    amplitude = amplitude_for_snr_db(snr_db, config.noise_std)

    _, clean = generate_sinusoid(
        fs_hz=config.fs_hz,
        duration_s=config.duration_s,
        frequency_hz=config.signal_hz,
        amplitude=amplitude,
    )

    noise = rng.normal(loc=0.0, scale=config.noise_std, size=clean.shape)
    return clean + noise


def apply_processing_chain(signal: np.ndarray, design: ChainDesign) -> tuple[np.ndarray, float]:
    """
    Apply optional filtering and windowing.

    Parameters
    ----------
    signal:
        Input time-domain signal.
    design:
        Processing-chain design.

    Returns
    -------
    tuple[np.ndarray, float]
        Processed signal and coherent gain.
    """
    x = signal

    if design.chain.filter_type == "fir_lowpass":
        if design.fir_b is None:
            raise ValueError("FIR chain is missing coefficients.")
        x = apply_filter(x, design.fir_b)

    elif design.chain.filter_type == "iir_lowpass":
        if design.iir_b is None or design.iir_a is None:
            raise ValueError("IIR chain is missing coefficients.")
        x = apply_filter(x, design.iir_b, design.iir_a)

    windowed, window = apply_window(x, design.chain.window_name)
    return windowed, coherent_gain(window)


def compute_spectrum_at_cut(
    signal: np.ndarray,
    design: ChainDesign,
    config: TrialConfig,
) -> SpectrumAtCut:
    """
    Compute corrected FFT spectrum and statistic at the expected Doppler bin.

    Parameters
    ----------
    signal:
        Raw time-domain trial.
    design:
        Processing-chain design.
    config:
        Trial configuration.

    Returns
    -------
    SpectrumAtCut
        Spectrum and detector cell-under-test values.
    """
    processed, cg = apply_processing_chain(signal, design)

    freq_hz, magnitude = compute_single_sided_fft(
        processed,
        fs_hz=config.fs_hz,
        remove_dc=True,
    )

    corrected_magnitude = magnitude / cg
    magnitude_db = magnitude_to_db(corrected_magnitude)
    power = corrected_magnitude**2

    cut_index = int(np.argmin(np.abs(freq_hz - config.signal_hz)))

    return SpectrumAtCut(
        freq_hz=freq_hz,
        magnitude_db=magnitude_db,
        power=power,
        cut_index=cut_index,
        cut_frequency_hz=float(freq_hz[cut_index]),
        cut_magnitude_db=float(magnitude_db[cut_index]),
        cut_power=float(power[cut_index]),
    )


def ca_cfar_alpha_from_pfa(target_pfa: float, n_reference_cells: int) -> float:
    """
    Compute the CA-CFAR multiplier for exponential noise power.

    Formula
    -------
        alpha = N * (Pfa^(-1/N) - 1)

    where N is the number of reference cells.

    Parameters
    ----------
    target_pfa:
        Desired probability of false alarm per cell under test.
    n_reference_cells:
        Number of reference cells used to estimate noise power.

    Returns
    -------
    float
        CA-CFAR threshold multiplier.
    """
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be in (0, 1).")
    if n_reference_cells < 1:
        raise ValueError("n_reference_cells must be positive.")

    n = float(n_reference_cells)
    return float(n * (target_pfa ** (-1.0 / n) - 1.0))


def cfar_reference_cells(
    power: np.ndarray,
    cut_index: int,
    *,
    guard_cells: int,
    reference_cells: int,
) -> np.ndarray:
    """
    Extract CA-CFAR reference cells around the cell under test.

    Parameters
    ----------
    power:
        Linear power spectrum.
    cut_index:
        Cell-under-test index.
    guard_cells:
        Guard cells excluded on each side of the CUT.
    reference_cells:
        Reference cells requested on each side.

    Returns
    -------
    np.ndarray
        Reference-cell power values.
    """
    x = np.asarray(power, dtype=float)

    left_start = max(0, cut_index - guard_cells - reference_cells)
    left_stop = max(0, cut_index - guard_cells)
    right_start = min(x.size, cut_index + guard_cells + 1)
    right_stop = min(x.size, cut_index + guard_cells + reference_cells + 1)

    refs = np.concatenate([x[left_start:left_stop], x[right_start:right_stop]])

    if refs.size < 2:
        raise ValueError("Not enough reference cells for CA-CFAR.")
    return refs


def ca_cfar_threshold_power(
    power: np.ndarray,
    cut_index: int,
    *,
    target_pfa: float,
    guard_cells: int,
    reference_cells: int,
) -> float:
    """
    Compute CA-CFAR threshold in linear power.

    All CFAR arithmetic is intentionally done in linear power. Converting to dB
    before averaging would produce an invalid threshold.
    """
    refs = cfar_reference_cells(
        power,
        cut_index,
        guard_cells=guard_cells,
        reference_cells=reference_cells,
    )
    alpha = ca_cfar_alpha_from_pfa(target_pfa, refs.size)
    return float(alpha * np.mean(refs))


def fixed_threshold_from_noise(
    *,
    design: ChainDesign,
    config: TrialConfig,
    target_pfa: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    Estimate fixed-threshold value from H0 CUT statistics.

    Returns
    -------
    tuple[float, np.ndarray]
        Threshold in dB and H0 CUT statistics in dB.
    """
    noise_stats_db = np.empty(n_trials, dtype=float)

    for idx in range(n_trials):
        noise = generate_noise_only_signal(config, rng)
        spectrum = compute_spectrum_at_cut(noise, design, config)
        noise_stats_db[idx] = spectrum.cut_magnitude_db

    threshold_db = float(np.quantile(noise_stats_db, 1.0 - target_pfa))
    return threshold_db, noise_stats_db


def first_snr_reaching_target(
    snr_db: np.ndarray,
    pd: np.ndarray,
    target_pd: float,
) -> float | None:
    """
    Estimate the SNR required to reach the target Pd.

    This function uses linear interpolation between adjacent SNR samples instead
    of returning the first sampled point above target. That distinction matters:
    the sampled crossing is grid-dependent, while the interpolated crossing is a
    useful engineering estimate.

    Returns
    -------
    float | None
        Interpolated SNR where Pd reaches target_pd. None if the target is not
        reached within the configured sweep.
    """
    snr = np.asarray(snr_db, dtype=float)
    prob = np.asarray(pd, dtype=float)

    order = np.argsort(snr)
    snr = snr[order]
    prob = prob[order]

    if np.all(prob < target_pd):
        return None

    if prob[0] >= target_pd:
        return float(snr[0])

    for idx in range(1, len(snr)):
        p0 = prob[idx - 1]
        p1 = prob[idx]

        if p0 <= target_pd <= p1:
            s0 = snr[idx - 1]
            s1 = snr[idx]

            if np.isclose(p1, p0):
                return float(s1)

            frac = (target_pd - p0) / (p1 - p0)
            return float(s0 + frac * (s1 - s0))

    return None
    

def evaluate_fixed_threshold_detector(
    *,
    design: ChainDesign,
    config: TrialConfig,
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> DetectorMetrics:
    """
    Evaluate fixed-threshold detection for one processing chain.
    """
    threshold_db, noise_stats_db = fixed_threshold_from_noise(
        design=design,
        config=config,
        target_pfa=args.target_pfa,
        n_trials=args.n_trials,
        rng=rng,
    )

    pd_values = np.empty_like(snr_values_db, dtype=float)

    for idx, snr_db in enumerate(snr_values_db):
        signal_stats_db = np.empty(args.n_trials, dtype=float)

        for trial_idx in range(args.n_trials):
            signal = generate_signal_present_signal(
                config,
                snr_db=float(snr_db),
                rng=rng,
            )
            spectrum = compute_spectrum_at_cut(signal, design, config)
            signal_stats_db[trial_idx] = spectrum.cut_magnitude_db

        pd_values[idx] = estimate_detection_probability(signal_stats_db, threshold=threshold_db)

    empirical_pfa = estimate_false_alarm_rate(noise_stats_db, threshold=threshold_db)

    return DetectorMetrics(
        chain_name=design.chain.name,
        detector="fixed_threshold",
        filter_type=design.chain.filter_type,
        window_name=design.chain.window_name,
        cost_units=design.chain.cost_units,
        empirical_pfa=empirical_pfa,
        threshold_db=threshold_db,
        mean_noise_stat_db=float(np.mean(noise_stats_db)),
        snr_db=snr_values_db.copy(),
        pd=pd_values,
        snr_for_target_pd_db=first_snr_reaching_target(snr_values_db, pd_values, args.target_pd),
        max_pd=float(np.max(pd_values)),
    )


def evaluate_ca_cfar_detector(
    *,
    design: ChainDesign,
    config: TrialConfig,
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> DetectorMetrics:
    """
    Evaluate CA-CFAR detection for one processing chain.

    The CA-CFAR threshold is computed from local reference cells around the
    expected Doppler bin for each trial.
    """
    noise_decisions = np.empty(args.n_trials, dtype=bool)
    noise_stats_db = np.empty(args.n_trials, dtype=float)
    noise_thresholds_db = np.empty(args.n_trials, dtype=float)

    for idx in range(args.n_trials):
        noise = generate_noise_only_signal(config, rng)
        spectrum = compute_spectrum_at_cut(noise, design, config)
        threshold_power = ca_cfar_threshold_power(
            spectrum.power,
            spectrum.cut_index,
            target_pfa=args.target_pfa,
            guard_cells=args.cfar_guard_cells,
            reference_cells=args.cfar_reference_cells,
        )
        threshold_db = float(10.0 * np.log10(threshold_power + 1e-300))

        noise_decisions[idx] = spectrum.cut_power > threshold_power
        noise_stats_db[idx] = spectrum.cut_magnitude_db
        noise_thresholds_db[idx] = threshold_db

    pd_values = np.empty_like(snr_values_db, dtype=float)

    for idx, snr_db in enumerate(snr_values_db):
        signal_decisions = np.empty(args.n_trials, dtype=bool)

        for trial_idx in range(args.n_trials):
            signal = generate_signal_present_signal(
                config,
                snr_db=float(snr_db),
                rng=rng,
            )
            spectrum = compute_spectrum_at_cut(signal, design, config)
            threshold_power = ca_cfar_threshold_power(
                spectrum.power,
                spectrum.cut_index,
                target_pfa=args.target_pfa,
                guard_cells=args.cfar_guard_cells,
                reference_cells=args.cfar_reference_cells,
            )
            signal_decisions[trial_idx] = spectrum.cut_power > threshold_power

        pd_values[idx] = float(np.mean(signal_decisions))

    empirical_pfa = float(np.mean(noise_decisions))

    return DetectorMetrics(
        chain_name=design.chain.name,
        detector="ca_cfar",
        filter_type=design.chain.filter_type,
        window_name=design.chain.window_name,
        cost_units=design.chain.cost_units + 0.7,
        empirical_pfa=empirical_pfa,
        threshold_db=float(np.mean(noise_thresholds_db)),
        mean_noise_stat_db=float(np.mean(noise_stats_db)),
        snr_db=snr_values_db.copy(),
        pd=pd_values,
        snr_for_target_pd_db=first_snr_reaching_target(snr_values_db, pd_values, args.target_pd),
        max_pd=float(np.max(pd_values)),
    )


def run_tradeoff_analysis(args: argparse.Namespace) -> list[DetectorMetrics]:
    """
    Run all Stage 05 chain/detector evaluations.
    """
    validate_configuration(args)

    config = build_trial_config(args)
    snr_values_db = np.array(args.snr_db, dtype=float)
    rng = np.random.default_rng(args.seed)

    metrics: list[DetectorMetrics] = []

    for chain in default_processing_chains():
        design = design_processing_chain(chain, args)

        metrics.append(
            evaluate_fixed_threshold_detector(
                design=design,
                config=config,
                snr_values_db=snr_values_db,
                args=args,
                rng=rng,
            )
        )

        metrics.append(
            evaluate_ca_cfar_detector(
                design=design,
                config=config,
                snr_values_db=snr_values_db,
                args=args,
                rng=rng,
            )
        )

    return metrics


def write_metrics_csv(metrics: list[DetectorMetrics], args: argparse.Namespace) -> Path:
    """
    Write the Stage 05 numerical summary to CSV.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "05_pipeline_metrics.csv"

    fieldnames = [
        "chain_name",
        "detector",
        "filter_type",
        "window_name",
        "cost_units",
        "empirical_pfa",
        "threshold_db",
        "mean_noise_stat_db",
        "snr_for_target_pd_db",
        "max_pd",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in metrics:
            writer.writerow(
                {
                    "chain_name": row.chain_name,
                    "detector": row.detector,
                    "filter_type": row.filter_type,
                    "window_name": row.window_name,
                    "cost_units": f"{row.cost_units:.3f}",
                    "empirical_pfa": f"{row.empirical_pfa:.6f}",
                    "threshold_db": f"{row.threshold_db:.6f}",
                    "mean_noise_stat_db": f"{row.mean_noise_stat_db:.6f}",
                    "snr_for_target_pd_db": (
                        ""
                        if row.snr_for_target_pd_db is None
                        else f"{row.snr_for_target_pd_db:.6f}"
                    ),
                    "max_pd": f"{row.max_pd:.6f}",
                }
            )

    return output_path


def plot_pd_curves(
    rows: list[DetectorMetrics],
    args: argparse.Namespace,
    *,
    detector_label: str,
    output_name: str,
) -> Path:
    """
    Plot Pd vs SNR for one detector family.

    Parameters
    ----------
    rows:
        Metrics for the detector family.
    args:
        Runtime configuration.
    detector_label:
        Figure title detector label.
    output_name:
        Output filename.

    Returns
    -------
    Path
        Generated figure path.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_name

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for row in rows:
        ax.plot(row.snr_db, row.pd, marker="o", linewidth=2.0, label=row.chain_name)

    ax.axhline(
        args.target_pd,
        linestyle="--",
        linewidth=1.5,
        label=f"Target Pd={args.target_pd:.2f}",
    )

    ax.set_title(f"{detector_label}: Pd vs SNR")
    ax.set_xlabel("Input SNR [dB]")
    ax.set_ylabel("Empirical Pd")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_false_alarm_control(metrics: list[DetectorMetrics], args: argparse.Namespace) -> Path:
    """
    Plot empirical Pfa for fixed-threshold and CA-CFAR detectors.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "05_false_alarm_control.png"

    chains = default_processing_chains()
    chain_names = [chain.name for chain in chains]
    x = np.arange(len(chain_names))
    width = 0.36

    fixed_pfa = [
        next(m.empirical_pfa for m in metrics if m.chain_name == name and m.detector == "fixed_threshold")
        for name in chain_names
    ]
    cfar_pfa = [
        next(m.empirical_pfa for m in metrics if m.chain_name == name and m.detector == "ca_cfar")
        for name in chain_names
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.bar(x - width / 2.0, fixed_pfa, width, label="Fixed threshold")
    ax.bar(x + width / 2.0, cfar_pfa, width, label="CA-CFAR")
    ax.axhline(args.target_pfa, linestyle="--", linewidth=1.6, label=f"Target Pfa={args.target_pfa:.2f}")

    ax.set_title("False-Alarm Control")
    ax.set_ylabel("Empirical Pfa")
    ax.set_xticks(x)
    ax.set_xticklabels(chain_names, rotation=25, ha="right")
    ax.set_ylim(0.0, max(0.12, 1.3 * max(max(fixed_pfa), max(cfar_pfa), args.target_pfa)))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_snr_requirement_summary(metrics: list[DetectorMetrics], args: argparse.Namespace) -> Path:
    """
    Plot interpolated SNR required to reach target Pd for each processing chain.

    This figure is intended to summarize engineering impact. It uses the
    fixed-threshold detector because that detector has explicitly controlled
    Monte Carlo thresholding and is directly comparable across chains.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "05_snr_requirement_summary.png"

    fixed_rows = [m for m in metrics if m.detector == "fixed_threshold"]
    chain_names = [m.chain_name for m in fixed_rows]
    values = [
        np.nan if m.snr_for_target_pd_db is None else m.snr_for_target_pd_db
        for m in fixed_rows
    ]

    x = np.arange(len(chain_names))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bars = ax.bar(x, values)

    ax.set_title(f"Interpolated SNR Required for Pd ≥ {args.target_pd:.2f}")
    ax.set_ylabel("Input SNR [dB]")
    ax.set_xticks(x)
    ax.set_xticklabels(chain_names, rotation=25, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    finite_values = [v for v in values if not np.isnan(v)]
    if finite_values:
        y_min = min(finite_values) - 1.5
        y_max = max(finite_values) + 1.5
        if np.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        ax.set_ylim(y_min, y_max)

    for bar, value in zip(bars, values):
        text = "not reached" if np.isnan(value) else f"{value:.2f} dB"
        y = 0.0 if np.isnan(value) else value
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2.0, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path
    

def generate_outputs(metrics: list[DetectorMetrics], args: argparse.Namespace) -> list[Path]:
    """
    Generate Stage 05 figures and CSV artifacts.
    """
    fixed_rows = [m for m in metrics if m.detector == "fixed_threshold"]
    cfar_rows = [m for m in metrics if m.detector == "ca_cfar"]

    paths = [
        plot_pd_curves(
            fixed_rows,
            args,
            detector_label="Fixed Threshold",
            output_name="05_fixed_threshold_pd_vs_snr.png",
        ),
        plot_pd_curves(
            cfar_rows,
            args,
            detector_label="CA-CFAR",
            output_name="05_ca_cfar_pd_vs_snr.png",
        ),
        plot_false_alarm_control(metrics, args),
        plot_snr_requirement_summary(metrics, args),
        write_metrics_csv(metrics, args),
    ]

    if args.show:
        for path in paths:
            if path.suffix.lower() != ".png":
                continue
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10.0, 6.0))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()

    return paths


def print_summary(metrics: list[DetectorMetrics], args: argparse.Namespace) -> None:
    """
    Print compact engineering summary.
    """
    print("\n[OK] Stage 05 system trade-off summary")
    for row in metrics:
        snr_req = (
            "not reached"
            if row.snr_for_target_pd_db is None
            else f"{row.snr_for_target_pd_db:.1f} dB"
        )
        print(
            "{chain:<27} | {det:<15} | Pfa={pfa:>6.3f} | "
            "Max Pd={maxpd:>5.3f} | SNR@Pd≥{target:.2f}: {snr}".format(
                chain=row.chain_name,
                det=row.detector,
                pfa=row.empirical_pfa,
                maxpd=row.max_pd,
                target=args.target_pd,
                snr=snr_req,
            )
        )


def print_generated_artifacts(paths: list[Path]) -> None:
    """
    Print generated artifact paths relative to the repository root.
    """
    print("\n[OK] Generated Stage 05 artifacts:")
    for path in paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")


def main() -> int:
    """
    Execute Stage 05.
    """
    args = parse_args()

    metrics = run_tradeoff_analysis(args)
    print_summary(metrics, args)

    paths = generate_outputs(metrics, args)
    print_generated_artifacts(paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
