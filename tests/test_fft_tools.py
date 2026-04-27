"""
test_fft_tools.py

Unit tests for src.fft_tools.

Purpose
-------
Validate FFT and spectral detectability invariants used by the executable
pipeline stages.

Engineering contracts checked here:

- a bin-centered sinusoid produces a peak at the correct FFT bin
- single-sided FFT scaling preserves sinusoidal peak amplitude
- linear magnitude to dB conversion is numerically correct
- noise-floor estimation excludes the protected peak region
- detectability is peak level minus noise floor
"""

import numpy as np

from src.fft_tools import (
    compute_detectability_db,
    compute_single_sided_fft,
    detect_spectral_peak,
    estimate_noise_floor,
    magnitude_to_db,
)
from src.signals import generate_sinusoid


def test_single_sided_fft_detects_correct_bin_centered_frequency() -> None:
    """
    A sinusoid exactly centered on an FFT bin should be detected at that bin.

    This protects the frequency-axis and FFT-scaling conventions used throughout
    the repo.
    """
    _, signal = generate_sinusoid(
        fs_hz=1000.0,
        duration_s=1.0,
        frequency_hz=50.0,
        amplitude=1.0,
    )

    freq_hz, magnitude = compute_single_sided_fft(signal, fs_hz=1000.0)
    peak = detect_spectral_peak(
        freq_hz,
        magnitude,
        search_min_hz=45.0,
        search_max_hz=55.0,
    )

    assert np.isclose(peak.frequency_hz, 50.0)
    assert np.isclose(peak.magnitude, 1.0, atol=1e-12)


def test_magnitude_to_db_returns_known_reference_values() -> None:
    """
    Validate standard amplitude-to-dB conversion:

    - 1.0  ->   0 dB
    - 10.0 ->  20 dB
    - 0.1  -> -20 dB
    """
    magnitude = np.array([1.0, 10.0, 0.1])
    expected_db = np.array([0.0, 20.0, -20.0])

    measured_db = magnitude_to_db(magnitude)

    assert np.allclose(measured_db, expected_db)


def test_noise_floor_estimation_excludes_peak_guard_region() -> None:
    """
    A large spectral peak should not contaminate the estimated noise floor.

    The peak bin and guard bins are excluded before computing the median floor.
    """
    spectrum_db = np.full(101, -40.0)
    spectrum_db[50] = 0.0

    estimated_floor_db = estimate_noise_floor(
        spectrum_db,
        exclude_index=50,
        guard_bins=3,
    )

    assert np.isclose(estimated_floor_db, -40.0)


def test_detectability_is_peak_minus_noise_floor() -> None:
    """
    Detectability is defined as peak magnitude above estimated noise floor.
    """
    detectability_db = compute_detectability_db(
        peak_magnitude_db=-5.0,
        noise_floor_db=-35.0,
    )

    assert np.isclose(detectability_db, 30.0)
