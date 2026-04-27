"""
test_detection.py

Unit tests for src.detection.

Purpose
-------
Validate detector primitives used by Stage 04 and Stage 05.

Engineering contracts checked here:

- scalar threshold detection counts decisions correctly
- empirical Pfa is the fraction of H0 statistics above threshold
- empirical Pd is the fraction of H1 statistics above threshold
- Doppler peak detector reports the correct peak, threshold, and margin
"""

import numpy as np

from src.detection import (
    doppler_peak_detector,
    estimate_detection_probability,
    estimate_false_alarm_rate,
    threshold_detector,
)


def test_threshold_detector_counts_expected_positive_decisions() -> None:
    """
    Threshold detector should mark values strictly greater than threshold.

    This convention is used consistently across Pd/Pfa estimation.
    """
    statistic = np.array([0.1, 0.5, 0.9, 1.3])

    result = threshold_detector(statistic, threshold=0.8)

    assert result.n_detections == 2
    assert np.isclose(result.detection_fraction, 0.5)
    assert result.detections.tolist() == [False, False, True, True]


def test_false_alarm_rate_matches_fraction_of_noise_statistics_above_threshold() -> None:
    """
    Empirical Pfa is the fraction of H0 statistics above threshold.
    """
    noise_statistic = np.array([0.0, 0.2, 0.4, 1.2, 1.5])

    pfa = estimate_false_alarm_rate(noise_statistic, threshold=1.0)

    assert np.isclose(pfa, 0.4)


def test_detection_probability_matches_fraction_of_signal_statistics_above_threshold() -> None:
    """
    Empirical Pd is the fraction of H1 statistics above threshold.
    """
    signal_statistic = np.array([0.2, 1.1, 1.4, 1.8, 2.0])

    pd = estimate_detection_probability(signal_statistic, threshold=1.0)

    assert np.isclose(pd, 0.8)


def test_doppler_peak_detector_reports_peak_threshold_and_margin() -> None:
    """
    Doppler peak detector should identify the strongest in-band peak and compare
    it against a noise-floor-relative threshold.
    """
    freq_hz = np.linspace(0.0, 100.0, 101)
    magnitude_db = np.full_like(freq_hz, -40.0)
    magnitude_db[50] = -10.0

    result = doppler_peak_detector(
        freq_hz,
        magnitude_db,
        search_min_hz=45.0,
        search_max_hz=55.0,
        threshold_offset_db=6.0,
    )

    assert result.detected is True
    assert np.isclose(result.peak_frequency_hz, 50.0)
    assert np.isclose(result.noise_floor_db, -40.0)
    assert np.isclose(result.threshold_db, -34.0)
    assert np.isclose(result.margin_db, 24.0)
