"""
test_windows.py

Unit tests for src.windows.

Purpose
-------
Validate windowing metrics used by Stage 03 and Stage 05.

Engineering contracts checked here:

- rectangular window is the coherent-gain and ENBW reference
- Hann window metrics match expected textbook values approximately
- Blackman has larger ENBW than Hann
- applying a window preserves signal shape while changing sample values
"""

import numpy as np
import pytest

from src.windows import (
    apply_window,
    coherent_gain,
    equivalent_noise_bandwidth,
    get_window,
)


def test_rectangular_window_is_reference_for_gain_and_enbw() -> None:
    """
    Rectangular window should have coherent gain 1 and ENBW 1 bin.

    This is the baseline against which other windows are compared.
    """
    window = get_window("rectangular", 1024)

    assert np.isclose(coherent_gain(window), 1.0)
    assert np.isclose(equivalent_noise_bandwidth(window), 1.0)


def test_hann_window_metrics_match_expected_values() -> None:
    """
    Hann window has coherent gain near 0.5 and ENBW near 1.5 bins.

    A large sample count is used to reduce finite-length approximation error.
    """
    window = get_window("hann", 4096)

    assert np.isclose(coherent_gain(window), 0.5, atol=5e-4)
    assert np.isclose(equivalent_noise_bandwidth(window), 1.5, atol=2e-3)


def test_blackman_window_has_larger_enbw_than_hann() -> None:
    """
    Blackman suppresses sidelobes more aggressively but pays with higher ENBW.
    """
    hann = get_window("hann", 4096)
    blackman = get_window("blackman", 4096)

    assert equivalent_noise_bandwidth(blackman) > equivalent_noise_bandwidth(hann)


def test_apply_window_preserves_shape_and_changes_values() -> None:
    """
    Applying a non-rectangular window should preserve array shape and change values.
    """
    signal = np.ones(128)

    windowed, window = apply_window(signal, "hann")

    assert windowed.shape == signal.shape
    assert window.shape == signal.shape
    assert not np.allclose(windowed, signal)


def test_apply_rectangular_window_preserves_signal_values() -> None:
    """
    Applying a rectangular window should leave the signal unchanged.

    This protects the reference behavior used to compare non-rectangular
    windows against the no-apodization case.
    """
    signal = np.array([1.0, -2.0, 3.5, 0.0, -1.25], dtype=float)

    windowed, window = apply_window(signal, "rectangular")

    assert windowed.shape == signal.shape
    assert window.shape == signal.shape
    assert np.allclose(window, np.ones_like(signal))
    assert np.allclose(windowed, signal)


def test_coherent_gain_rejects_non_1d_window() -> None:
    """
    Coherent gain should reject non-1D inputs.

    Window metrics in this repo are defined for one-dimensional window vectors,
    not matrices or higher-dimensional arrays.
    """
    window = np.ones((8, 8), dtype=float)

    with pytest.raises(ValueError, match="window must be one-dimensional."):
        coherent_gain(window)
