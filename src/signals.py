"""
signals.py

Signal generation utilities for the radar DSP pipeline.

Pipeline role
-------------
This module provides deterministic and noisy baseband/test signals used by the
pipeline stages. It intentionally contains no plotting, command-line interface,
or file-system side effects.

Used by
-------
- scripts/01_signals_noise_fft.py
- scripts/02_fir_iir_filters.py
- scripts/03_windowing_leakage.py
- scripts/04_detection_doppler.py
- scripts/05_system_tradeoffs.py

Design contract
---------------
- Generate signals only.
- Keep random behavior reproducible through explicit RNG handling.
- Do not perform FFT, plotting, detection, or filtering here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_sinusoid(
    *,
    fs_hz: float,
    duration_s: float,
    frequency_hz: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a real-valued sinusoidal signal.

    Parameters
    ----------
    fs_hz:
        Sampling frequency in hertz.
    duration_s:
        Signal duration in seconds.
    frequency_hz:
        Sinusoid frequency in hertz.
    amplitude:
        Peak amplitude of the sinusoid.
    phase_rad:
        Initial phase in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time vector and sinusoidal signal.

    Raises
    ------
    ValueError
        If sampling frequency, duration, or amplitude are invalid.
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")

    n_samples = int(round(fs_hz * duration_s))
    if n_samples < 2:
        raise ValueError("Signal must contain at least two samples.")

    t = np.arange(n_samples, dtype=float) / fs_hz
    x = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)

    return t, x


def add_awgn_for_snr(
    signal: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add additive white Gaussian noise to achieve a target time-domain SNR.

    The SNR convention used here is:

        SNR = mean(signal^2) / mean(noise^2)

    This is a controlled engineering input for later spectral detectability
    analysis. It is not a received radar equation model.

    Parameters
    ----------
    signal:
        Input signal array.
    snr_db:
        Target signal-to-noise ratio in decibels.
    rng:
        Optional NumPy random generator. If omitted, a new default generator is
        used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Noisy signal and generated noise component.

    Raises
    ------
    ValueError
        If the input signal is empty or has zero power.
    """
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        raise ValueError("signal must not be empty.")

    signal_power = float(np.mean(x**2))
    if signal_power <= 0.0:
        raise ValueError("signal power must be positive to define SNR.")

    generator = rng if rng is not None else np.random.default_rng()

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = generator.normal(loc=0.0, scale=np.sqrt(noise_power), size=x.shape)

    return x + noise, noise


def generate_noisy_sinusoid(
    *,
    fs_hz: float,
    duration_s: float,
    frequency_hz: float,
    snr_db: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
    seed: int | None = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sinusoid corrupted by additive white Gaussian noise.

    This is the standard source signal for the first DSP pipeline stage:
    signal buried in noise → FFT spectral peak → detectability metric.

    Parameters
    ----------
    fs_hz:
        Sampling frequency in hertz.
    duration_s:
        Signal duration in seconds.
    frequency_hz:
        Sinusoid frequency in hertz.
    snr_db:
        Target time-domain SNR in decibels.
    amplitude:
        Peak amplitude of the sinusoid.
    phase_rad:
        Initial phase in radians.
    seed:
        Random seed used for reproducible AWGN generation. Pass None for
        non-deterministic behavior.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Time vector, noisy signal, clean signal, and noise component.
    """
    t, clean = generate_sinusoid(
        fs_hz=fs_hz,
        duration_s=duration_s,
        frequency_hz=frequency_hz,
        amplitude=amplitude,
        phase_rad=phase_rad,
    )

    rng = np.random.default_rng(seed)
    noisy, noise = add_awgn_for_snr(clean, snr_db=snr_db, rng=rng)

    return t, noisy, clean, noise