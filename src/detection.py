"""
detection.py

Detection utilities for the radar DSP pipeline.

Purpose
-------
Provide reusable thresholding and Doppler peak detection primitives used by the
executable pipeline stages.

Pipeline role
-------------
Used by:
    scripts/04_detection_doppler.py
    scripts/05_system_tradeoffs.py

Interacts with:
    src.signals
    src.fft_tools
    src.windows

Design contract
---------------
- No plotting.
- No CLI.
- No file-system side effects.
- No random signal generation.
- Functions operate on arrays and return explicit numerical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdDetectionResult:
    """
    Result of scalar threshold detection.

    Attributes
    ----------
    detections:
        Boolean detection decisions.
    threshold:
        Threshold applied to the statistic.
    n_detections:
        Number of positive detections.
    detection_fraction:
        Fraction of samples above threshold.
    """

    detections: np.ndarray
    threshold: float
    n_detections: int
    detection_fraction: float


@dataclass(frozen=True)
class DopplerPeakDetectionResult:
    """
    Result of Doppler/spectral peak detection.

    Attributes
    ----------
    detected:
        True if the peak exceeds the requested threshold.
    peak_frequency_hz:
        Frequency of the detected peak.
    peak_magnitude_db:
        Peak magnitude in dB.
    noise_floor_db:
        Estimated local/global noise floor in dB.
    threshold_db:
        Detection threshold in dB.
    margin_db:
        Peak level minus threshold in dB.
    bin_index:
        Index of the detected frequency bin.
    """

    detected: bool
    peak_frequency_hz: float
    peak_magnitude_db: float
    noise_floor_db: float
    threshold_db: float
    margin_db: float
    bin_index: int


def threshold_detector(
    statistic: np.ndarray,
    *,
    threshold: float,
) -> ThresholdDetectionResult:
    """
    Apply a scalar threshold detector.

    Parameters
    ----------
    statistic:
        Detection statistic values.
    threshold:
        Detection threshold. Samples strictly greater than this value are
        declared detections.

    Returns
    -------
    ThresholdDetectionResult
        Detection decisions and summary statistics.

    Raises
    ------
    ValueError
        If statistic is empty.
    """
    values = np.asarray(statistic, dtype=float)
    if values.size == 0:
        raise ValueError("statistic must not be empty.")

    detections = values > threshold
    n_detections = int(np.count_nonzero(detections))
    detection_fraction = float(n_detections / values.size)

    return ThresholdDetectionResult(
        detections=detections,
        threshold=float(threshold),
        n_detections=n_detections,
        detection_fraction=detection_fraction,
    )


def estimate_false_alarm_rate(
    noise_only_statistic: np.ndarray,
    *,
    threshold: float,
) -> float:
    """
    Estimate empirical probability of false alarm from noise-only statistics.

    Parameters
    ----------
    noise_only_statistic:
        Detection statistic under H0/noise-only conditions.
    threshold:
        Detection threshold.

    Returns
    -------
    float
        Empirical false-alarm probability.

    Notes
    -----
    This function returns probability per decision, not operational FAR per
    second. System-level FAR requires multiplying by the number of decisions per
    second.
    """
    result = threshold_detector(noise_only_statistic, threshold=threshold)
    return result.detection_fraction


def estimate_detection_probability(
    signal_present_statistic: np.ndarray,
    *,
    threshold: float,
) -> float:
    """
    Estimate empirical probability of detection from signal-present statistics.

    Parameters
    ----------
    signal_present_statistic:
        Detection statistic under H1/signal-present conditions.
    threshold:
        Detection threshold.

    Returns
    -------
    float
        Empirical detection probability.
    """
    result = threshold_detector(signal_present_statistic, threshold=threshold)
    return result.detection_fraction


def doppler_peak_detector(
    freq_hz: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    search_min_hz: float | None = None,
    search_max_hz: float | None = None,
    noise_floor_db: float | None = None,
    threshold_offset_db: float = 6.0,
) -> DopplerPeakDetectionResult:
    """
    Detect the strongest Doppler/spectral peak inside an optional search band.

    Parameters
    ----------
    freq_hz:
        Frequency axis in hertz.
    magnitude_db:
        Spectrum magnitude in dB.
    search_min_hz:
        Optional lower bound of the Doppler search band.
    search_max_hz:
        Optional upper bound of the Doppler search band.
    noise_floor_db:
        Optional externally estimated noise floor in dB. If None, the median
        of the non-peak spectrum is used.
    threshold_offset_db:
        Detection threshold above the estimated noise floor.

    Returns
    -------
    DopplerPeakDetectionResult
        Doppler peak detection decision and supporting metrics.

    Raises
    ------
    ValueError
        If inputs are inconsistent or the search band is empty.
    """
    f = np.asarray(freq_hz, dtype=float)
    y = np.asarray(magnitude_db, dtype=float)

    if f.size == 0 or y.size == 0:
        raise ValueError("freq_hz and magnitude_db must not be empty.")
    if f.shape != y.shape:
        raise ValueError("freq_hz and magnitude_db must have the same shape.")

    search_mask = np.ones(f.shape, dtype=bool)

    if search_min_hz is not None:
        search_mask &= f >= search_min_hz
    if search_max_hz is not None:
        search_mask &= f <= search_max_hz

    candidate_indices = np.flatnonzero(search_mask)
    if candidate_indices.size == 0:
        raise ValueError("Doppler search band contains no bins.")

    local_peak_index = int(np.argmax(y[candidate_indices]))
    peak_index = int(candidate_indices[local_peak_index])

    peak_frequency_hz = float(f[peak_index])
    peak_magnitude_db = float(y[peak_index])

    if noise_floor_db is None:
        floor_mask = np.ones(y.shape, dtype=bool)
        floor_mask[max(0, peak_index - 5): min(y.size, peak_index + 6)] = False
        if not np.any(floor_mask):
            raise ValueError("No bins available for noise-floor estimation.")
        estimated_floor_db = float(np.median(y[floor_mask]))
    else:
        estimated_floor_db = float(noise_floor_db)

    threshold_db = estimated_floor_db + float(threshold_offset_db)
    margin_db = peak_magnitude_db - threshold_db

    return DopplerPeakDetectionResult(
        detected=bool(margin_db > 0.0),
        peak_frequency_hz=peak_frequency_hz,
        peak_magnitude_db=peak_magnitude_db,
        noise_floor_db=estimated_floor_db,
        threshold_db=threshold_db,
        margin_db=float(margin_db),
        bin_index=peak_index,
    )