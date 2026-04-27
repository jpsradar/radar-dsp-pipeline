"""
test_signals.py

Unit tests for src.signals.

Purpose
-------
Validate the signal-generation invariants used throughout the radar DSP
pipeline.

These tests are not checking visual behavior or implementation details. They
check engineering contracts:

- sinusoid power follows the expected A²/2 relation
- AWGN generation is reproducible when an RNG seed is supplied
- generated noisy signals achieve the requested SNR within Monte Carlo tolerance
"""

import numpy as np

from src.signals import add_awgn_for_snr, generate_noisy_sinusoid, generate_sinusoid


def test_generate_sinusoid_power_matches_theoretical_average_power() -> None:
    """
    A real sinusoid with peak amplitude A has average power A²/2.

    This invariant is used later when converting requested SNR into controlled
    signal/noise experiments.
    """
    amplitude = 2.0
    expected_power = amplitude**2 / 2.0

    _, signal = generate_sinusoid(
        fs_hz=1000.0,
        duration_s=1.0,
        frequency_hz=50.0,
        amplitude=amplitude,
    )

    measured_power = float(np.mean(signal**2))

    assert np.isclose(measured_power, expected_power, rtol=1e-2)


def test_add_awgn_for_snr_is_reproducible_with_explicit_rng_seed() -> None:
    """
    Pipeline figures and Monte Carlo runs must be reproducible.

    This test verifies that AWGN generation is deterministic when the caller
    provides an explicitly seeded NumPy generator.
    """
    _, clean = generate_sinusoid(
        fs_hz=1000.0,
        duration_s=1.0,
        frequency_hz=50.0,
        amplitude=1.0,
    )

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    noisy_a, noise_a = add_awgn_for_snr(clean, snr_db=0.0, rng=rng_a)
    noisy_b, noise_b = add_awgn_for_snr(clean, snr_db=0.0, rng=rng_b)

    assert np.allclose(noisy_a, noisy_b)
    assert np.allclose(noise_a, noise_b)


def test_generate_noisy_sinusoid_achieves_requested_snr_within_tolerance() -> None:
    """
    The measured SNR should match the requested SNR within statistical tolerance.

    A longer signal is used here so the measured noise power is stable enough
    for a deterministic unit test.
    """
    requested_snr_db = 10.0

    _, noisy, clean, noise = generate_noisy_sinusoid(
        fs_hz=5000.0,
        duration_s=2.0,
        frequency_hz=50.0,
        snr_db=requested_snr_db,
        seed=123,
    )

    measured_snr_db = 10.0 * np.log10(np.mean(clean**2) / np.mean(noise**2))

    assert noisy.shape == clean.shape == noise.shape
    assert np.isclose(measured_snr_db, requested_snr_db, atol=0.35)
