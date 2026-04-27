"""
fft_tools.py

FFT and spectral detectability utilities for the radar DSP pipeline.

Pipeline role
-------------
This module converts time-domain signals into single-sided spectra and provides
basic spectral detectability measurements.

The core engineering question addressed here is:

    Can a deterministic signal component be distinguished from the surrounding
    spectral noise floor?

Used by
-------
- scripts/01_signals_noise_fft.py
- scripts/03_windowing_leakage.py
- scripts/04_detection_doppler.py
- scripts/05_system_tradeoffs.py

Design contract
---------------
- Perform FFT and spectral metric calculations only.
- Do not generate signals.
- Do not plot.
- Do not parse command-line arguments.
- Do not write files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralPeak:
    """
    Result of spectral peak detection.

    Attributes
    ----------
    frequency_hz:
        Frequency of the detected peak.
    magnitude:
        Linear magnitude of the detected peak.
    magnitude_db:
        Magnitude of the detected peak in decibels.
    bin_index:
        FFT bin index corresponding to the detected peak.
    """

    frequency_hz: float
    magnitude: float
    magnitude_db: float
    bin_index: int


def compute_single_sided_fft(
    signal: np.ndarray,
    *,
    fs_hz: float,
    remove_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a single-sided FFT magnitude spectrum for a real-valued signal.

    Parameters
    ----------
    signal:
        Input time-domain signal.
    fs_hz:
        Sampling frequency in hertz.
    remove_dc:
        If True, subtract the signal mean before computing the FFT.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequency axis in hertz and normalized single-sided magnitude spectrum.

    Notes
    -----
    The returned magnitude is normalized by the number of samples. Non-DC,
    non-Nyquist positive-frequency bins are doubled to preserve sinusoidal
    peak amplitude in the single-sided spectrum.

    Raises
    ------
    ValueError
        If input signal or sampling frequency are invalid.
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")

    x = np.asarray(signal, dtype=float)
    if x.size < 2:
        raise ValueError("signal must contain at least two samples.")

    if remove_dc:
        x = x - np.mean(x)

    n_samples = x.size
    spectrum = np.fft.rfft(x)
    freq_hz = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)

    magnitude = np.abs(spectrum) / n_samples

    if n_samples > 2:
        if n_samples % 2 == 0:
            magnitude[1:-1] *= 2.0
        else:
            magnitude[1:] *= 2.0

    return freq_hz, magnitude


def magnitude_to_db(
    magnitude: np.ndarray,
    *,
    floor_db: float = -240.0,
) -> np.ndarray:
    """
    Convert linear magnitude to decibels.

    Parameters
    ----------
    magnitude:
        Linear magnitude array.
    floor_db:
        Lower numerical floor in dB to avoid log of zero.

    Returns
    -------
    np.ndarray
        Magnitude in decibels.
    """
    mag = np.asarray(magnitude, dtype=float)

    if floor_db >= 0:
        raise ValueError("floor_db should be negative.")

    floor_linear = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(mag, floor_linear))


def estimate_noise_floor(
    magnitude_db: np.ndarray,
    *,
    exclude_index: int | None = None,
    guard_bins: int = 5,
    method: str = "median",
) -> float:
    """
    Estimate spectral noise floor in dB.

    Parameters
    ----------
    magnitude_db:
        Spectrum magnitude in decibels.
    exclude_index:
        Optional index around which bins are excluded from noise-floor
        estimation. Typically this is the detected signal peak.
    guard_bins:
        Number of bins to exclude on each side of exclude_index.
    method:
        Estimation method. Supported values: "median", "mean".

    Returns
    -------
    float
        Estimated noise floor in dB.

    Raises
    ------
    ValueError
        If the spectrum is empty, guard_bins is invalid, or method is unknown.
    """
    y = np.asarray(magnitude_db, dtype=float)
    if y.size == 0:
        raise ValueError("magnitude_db must not be empty.")
    if guard_bins < 0:
        raise ValueError("guard_bins must be non-negative.")

    mask = np.ones(y.shape, dtype=bool)

    if exclude_index is not None:
        if exclude_index < 0 or exclude_index >= y.size:
            raise ValueError("exclude_index is outside the spectrum.")
        start = max(0, exclude_index - guard_bins)
        stop = min(y.size, exclude_index + guard_bins + 1)
        mask[start:stop] = False

    samples = y[mask]
    if samples.size == 0:
        raise ValueError("No bins available for noise-floor estimation.")

    if method == "median":
        return float(np.median(samples))
    if method == "mean":
        return float(np.mean(samples))

    raise ValueError(f"Unsupported noise floor method: {method!r}")


def detect_spectral_peak(
    freq_hz: np.ndarray,
    magnitude: np.ndarray,
    *,
    search_min_hz: float | None = None,
    search_max_hz: float | None = None,
) -> SpectralPeak:
    """
    Detect the largest spectral peak within an optional search band.

    Parameters
    ----------
    freq_hz:
        Frequency axis in hertz.
    magnitude:
        Linear magnitude spectrum.
    search_min_hz:
        Optional lower search frequency.
    search_max_hz:
        Optional upper search frequency.

    Returns
    -------
    SpectralPeak
        Peak frequency, magnitude, magnitude in dB, and bin index.

    Raises
    ------
    ValueError
        If inputs are inconsistent or the search band is empty.
    """
    f = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(magnitude, dtype=float)

    if f.size == 0 or mag.size == 0:
        raise ValueError("freq_hz and magnitude must not be empty.")
    if f.shape != mag.shape:
        raise ValueError("freq_hz and magnitude must have the same shape.")

    mask = np.ones(f.shape, dtype=bool)

    if search_min_hz is not None:
        mask &= f >= search_min_hz
    if search_max_hz is not None:
        mask &= f <= search_max_hz

    candidate_indices = np.flatnonzero(mask)
    if candidate_indices.size == 0:
        raise ValueError("Peak search band contains no FFT bins.")

    local_index = int(np.argmax(mag[candidate_indices]))
    peak_index = int(candidate_indices[local_index])

    peak_mag = float(mag[peak_index])
    peak_mag_db = float(magnitude_to_db(np.array([peak_mag]))[0])

    return SpectralPeak(
        frequency_hz=float(f[peak_index]),
        magnitude=peak_mag,
        magnitude_db=peak_mag_db,
        bin_index=peak_index,
    )


def compute_detectability_db(
    peak_magnitude_db: float,
    noise_floor_db: float,
) -> float:
    """
    Compute spectral detectability as peak level above noise floor.

    Parameters
    ----------
    peak_magnitude_db:
        Detected peak magnitude in dB.
    noise_floor_db:
        Estimated noise floor in dB.

    Returns
    -------
    float
        Detectability in dB.

    Notes
    -----
    This is not Pd and not a formal detector statistic. It is an engineering
    visibility metric used to connect SNR, FFT processing, and later detection
    stages.
    """
    return float(peak_magnitude_db - noise_floor_db)