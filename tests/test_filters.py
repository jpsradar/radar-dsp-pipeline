"""
test_filters.py

Unit tests for src.filters.

Purpose
-------
Validate basic FIR/IIR low-pass filter behavior used in Stage 02 and Stage 05.

Engineering contracts checked here:

- passband gain remains near 0 dB
- stopband attenuation is meaningfully below passband
- filtering preserves signal length
- FIR and IIR design functions return usable coefficients
"""

import numpy as np

from src.filters import (
    apply_filter,
    compute_frequency_response,
    design_fir_lowpass,
    design_iir_lowpass,
)


def _response_at_frequency(freq_hz: np.ndarray, response_db: np.ndarray, target_hz: float) -> float:
    """
    Return response value nearest to a target frequency.

    Frequency response grids rarely land exactly on the target value, so tests
    should compare the nearest grid point instead of assuming exact alignment.
    """
    index = int(np.argmin(np.abs(freq_hz - target_hz)))
    return float(response_db[index])


def test_fir_lowpass_has_passband_gain_and_stopband_attenuation() -> None:
    """
    FIR low-pass design should keep low-frequency content and reject high-frequency content.

    The exact stopband value is not the point. The invariant is that the filter
    behaves like a real low-pass filter with meaningful attenuation.
    """
    b = design_fir_lowpass(
        cutoff_hz=100.0,
        fs_hz=1000.0,
        num_taps=101,
    )

    freq_hz, response_db = compute_frequency_response(
        b,
        fs_hz=1000.0,
        worN=4096,
    )

    passband_db = _response_at_frequency(freq_hz, response_db, target_hz=20.0)
    stopband_db = _response_at_frequency(freq_hz, response_db, target_hz=300.0)

    assert passband_db > -1.0
    assert stopband_db < -40.0


def test_iir_lowpass_has_passband_gain_and_stopband_attenuation() -> None:
    """
    Butterworth IIR low-pass design should preserve passband and attenuate stopband.

    The stopband threshold is looser than the FIR test because a fourth-order
    IIR is intentionally cheaper and less selective.
    """
    b, a = design_iir_lowpass(
        cutoff_hz=100.0,
        fs_hz=1000.0,
        order=4,
    )

    freq_hz, response_db = compute_frequency_response(
        b,
        a=a,
        fs_hz=1000.0,
        worN=4096,
    )

    passband_db = _response_at_frequency(freq_hz, response_db, target_hz=20.0)
    stopband_db = _response_at_frequency(freq_hz, response_db, target_hz=300.0)

    assert passband_db > -1.0
    assert stopband_db < -25.0


def test_apply_filter_preserves_input_signal_length() -> None:
    """
    Filtering should not change the number of samples.

    Later pipeline stages assume aligned input/output array sizes when comparing
    spectra and detectability metrics.
    """
    signal = np.ones(256)
    b = design_fir_lowpass(
        cutoff_hz=100.0,
        fs_hz=1000.0,
        num_taps=31,
    )

    filtered = apply_filter(signal, b)

    assert filtered.shape == signal.shape
