"""
windows.py

Windowing utilities for the radar DSP pipeline.

Purpose
-------
Provide reusable window functions and window metrics used to analyze spectral
leakage, coherent gain, and noise-bandwidth trade-offs.

Pipeline role
-------------
Used by:
    scripts/03_windowing_leakage.py
    scripts/05_system_tradeoffs.py

Interacts with:
    src.signals
    src.fft_tools

Design contract
---------------
- No plotting.
- No CLI.
- No file-system side effects.
- Only window generation/application and window metrics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


WindowName = Literal["rectangular", "hann", "hamming", "blackman"]


def get_window(window_name: WindowName, n_samples: int) -> np.ndarray:
    """
    Generate a deterministic window vector.

    Parameters
    ----------
    window_name:
        Window type. Supported values: "rectangular", "hann", "hamming",
        and "blackman".
    n_samples:
        Number of samples in the window.

    Returns
    -------
    np.ndarray
        Window coefficients.

    Raises
    ------
    ValueError
        If n_samples is invalid or the window type is unsupported.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    if window_name == "rectangular":
        return np.ones(n_samples, dtype=float)
    if window_name == "hann":
        return np.hanning(n_samples)
    if window_name == "hamming":
        return np.hamming(n_samples)
    if window_name == "blackman":
        return np.blackman(n_samples)

    raise ValueError(f"Unsupported window type: {window_name!r}")


def apply_window(signal: np.ndarray, window_name: WindowName) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a window to a one-dimensional signal.

    Parameters
    ----------
    signal:
        Input time-domain signal.
    window_name:
        Window type to apply.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Windowed signal and window coefficients.

    Raises
    ------
    ValueError
        If signal is not one-dimensional.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    window = get_window(window_name, x.size)
    return x * window, window


def coherent_gain(window: np.ndarray) -> float:
    """
    Compute coherent gain of a window.

    Coherent gain describes the amplitude loss applied to a sinusoid that is
    coherent with an FFT bin. It is required when comparing spectral peak
    amplitudes across different windows.

    Parameters
    ----------
    window:
        Window coefficients.

    Returns
    -------
    float
        Coherent gain.

    Raises
    ------
    ValueError
        If the window is empty.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        raise ValueError("window must not be empty.")

    return float(np.mean(w))


def equivalent_noise_bandwidth(window: np.ndarray) -> float:
    """
    Compute equivalent noise bandwidth (ENBW) in FFT bins.

    ENBW quantifies how much white-noise power passes through the window
    relative to an ideal rectangular bin. Higher ENBW generally means increased
    noise floor after windowing.

    Parameters
    ----------
    window:
        Window coefficients.

    Returns
    -------
    float
        Equivalent noise bandwidth in FFT bins.

    Raises
    ------
    ValueError
        If the window is empty or has zero coherent gain.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        raise ValueError("window must not be empty.")

    denominator = float(np.sum(w) ** 2)
    if denominator <= 0.0:
        raise ValueError("window has zero coherent gain.")

    return float(w.size * np.sum(w**2) / denominator)