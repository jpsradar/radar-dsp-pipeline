"""
filters.py

Signal filtering utilities for the radar DSP pipeline.

Purpose
-------
Provide deterministic, reusable filtering primitives to support analysis of:

- noise bandwidth reduction
- spectral shaping
- impact on detectability
- FIR vs IIR trade-offs

This module is intentionally limited to essential building blocks.
It is NOT a full DSP library.

Pipeline role
-------------
Used by:
    scripts/02_fir_iir_filters.py

Interacts with:
    src.signals      (signal generation)
    src.fft_tools    (spectral analysis)

Design principles
-----------------
- No plotting
- No CLI
- Deterministic behavior
- Explicit parameterization
- Compatible with both FIR and IIR workflows
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, firwin, freqz, lfilter


# ---------------------------------------------------------------------
# FIR DESIGN
# ---------------------------------------------------------------------
def design_fir_lowpass(
    cutoff_hz: float,
    fs_hz: float,
    num_taps: int,
    window: str = "hann",
) -> np.ndarray:
    """
    Design a low-pass FIR filter using the window method.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    fs_hz : float
        Sampling frequency in Hz.
    num_taps : int
        Number of FIR coefficients (filter length).
    window : str, optional
        Window function name (default: 'hann').

    Returns
    -------
    b : np.ndarray
        FIR filter coefficients.

    Notes
    -----
    - FIR filters have linear phase.
    - Increasing num_taps improves transition sharpness but increases cost.
    """
    if cutoff_hz <= 0 or cutoff_hz >= fs_hz / 2:
        raise ValueError("cutoff_hz must be within (0, fs/2)")

    if num_taps < 2:
        raise ValueError("num_taps must be >= 2")

    nyquist = fs_hz / 2.0
    normalized_cutoff = cutoff_hz / nyquist

    b = firwin(num_taps, normalized_cutoff, window=window)
    return b


# ---------------------------------------------------------------------
# IIR DESIGN
# ---------------------------------------------------------------------
def design_iir_lowpass(
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a low-pass Butterworth IIR filter.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    fs_hz : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default: 4).

    Returns
    -------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.

    Notes
    -----
    - Butterworth filters are maximally flat in passband.
    - IIR filters are computationally efficient but introduce phase distortion.
    """
    if cutoff_hz <= 0 or cutoff_hz >= fs_hz / 2:
        raise ValueError("cutoff_hz must be within (0, fs/2)")

    if order < 1:
        raise ValueError("order must be >= 1")

    nyquist = fs_hz / 2.0
    normalized_cutoff = cutoff_hz / nyquist

    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return b, a


# ---------------------------------------------------------------------
# FILTER APPLICATION
# ---------------------------------------------------------------------
def apply_filter(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply FIR or IIR filter to a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray, optional
        Denominator coefficients (for IIR). If None, FIR is assumed.

    Returns
    -------
    y : np.ndarray
        Filtered signal.

    Notes
    -----
    - Uses direct-form filtering (lfilter).
    - No zero-phase filtering (no filtfilt) to preserve causality.
    """
    if x.ndim != 1:
        raise ValueError("Input signal must be 1D")

    if a is None:
        a = np.array([1.0])

    y = lfilter(b, a, x)
    return y


# ---------------------------------------------------------------------
# FREQUENCY RESPONSE
# ---------------------------------------------------------------------
def compute_frequency_response(
    b: np.ndarray,
    a: np.ndarray | None = None,
    fs_hz: float = 1.0,
    worN: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency response of a filter.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray, optional
        Denominator coefficients. If None, FIR assumed.
    fs_hz : float, optional
        Sampling frequency (default: 1.0).
    worN : int, optional
        Number of frequency points.

    Returns
    -------
    freq_hz : np.ndarray
        Frequency axis in Hz.
    magnitude_db : np.ndarray
        Magnitude response in dB.

    Notes
    -----
    - Output is single-sided (0 to Nyquist).
    - Useful for analyzing bandwidth and attenuation.
    """
    if a is None:
        a = np.array([1.0])

    w, h = freqz(b, a, worN=worN)

    freq_hz = (w / (2 * np.pi)) * fs_hz
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)

    return freq_hz, magnitude_db