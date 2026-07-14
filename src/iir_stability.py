"""
iir_stability.py

Reusable utilities for IIR stability and numerical robustness experiments.

Purpose
-------
Provide deterministic building blocks for comparing:

- stable, near-unit-circle, and unstable pole radii
- direct-form and second-order-section implementations
- float32 and float64 numerical behavior

The module contains no plotting and no command-line interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, sos2tf, sosfilt


@dataclass(frozen=True)
class StabilityCase:
    """
    Configuration for one pole-radius experiment.

    Attributes
    ----------
    label:
        Human-readable case label.
    pole_radius:
        Magnitude of each conjugate pole pair in the z-plane.
    """

    label: str
    pole_radius: float


@dataclass(frozen=True)
class StabilityResult:
    """
    Numerical result for one IIR implementation.

    Attributes
    ----------
    case_label:
        Human-readable experiment label.
    implementation:
        Filtering structure name.
    dtype_name:
        Floating-point data type name.
    pole_radius:
        Pole radius used by the repeated resonator.
    output:
        Time-domain impulse response.
    peak_magnitude:
        Maximum finite absolute output value.
    final_magnitude:
        Absolute value of the final output sample.
    all_finite:
        Whether every output sample is finite.
    """

    case_label: str
    implementation: str
    dtype_name: str
    pole_radius: float
    output: np.ndarray
    peak_magnitude: float
    final_magnitude: float
    all_finite: bool


def build_repeated_resonator_sos(
    *,
    pole_radius: float,
    pole_angle_rad: float,
    section_count: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Build repeated second-order resonator sections.

    Parameters
    ----------
    pole_radius:
        Pole magnitude in the z-plane.
    pole_angle_rad:
        Conjugate-pole angle in radians.
    section_count:
        Number of repeated second-order sections.
    dtype:
        Floating-point dtype for coefficients.

    Returns
    -------
    np.ndarray
        SOS coefficient array with shape ``(section_count, 6)``.

    Raises
    ------
    ValueError
        If the pole radius, angle, or section count is invalid.
    TypeError
        If dtype is not float32 or float64.
    """
    if pole_radius <= 0.0:
        raise ValueError("pole_radius must be positive.")

    if not 0.0 < pole_angle_rad < np.pi:
        raise ValueError("pole_angle_rad must be within (0, pi) radians.")

    if section_count < 1:
        raise ValueError("section_count must be at least 1.")

    validated_dtype = np.dtype(dtype)
    if validated_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64.")

    denominator = np.array(
        [
            1.0,
            -2.0 * pole_radius * np.cos(pole_angle_rad),
            pole_radius**2,
        ],
        dtype=validated_dtype,
    )
    numerator = np.array([1.0, 0.0, 0.0], dtype=validated_dtype)

    section = np.concatenate((numerator, denominator))
    return np.tile(section, (section_count, 1))


def generate_unit_impulse(
    num_samples: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Generate a unit impulse.

    Parameters
    ----------
    num_samples:
        Output length in samples.
    dtype:
        Floating-point output dtype.

    Returns
    -------
    np.ndarray
        Unit impulse with shape ``(num_samples,)``.

    Raises
    ------
    ValueError
        If num_samples is not positive.
    TypeError
        If dtype is not float32 or float64.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    validated_dtype = np.dtype(dtype)
    if validated_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64.")

    impulse = np.zeros(num_samples, dtype=validated_dtype)
    impulse[0] = 1.0
    return impulse


def run_stability_case(
    *,
    case: StabilityCase,
    pole_angle_rad: float,
    section_count: int,
    num_samples: int,
    dtype: np.dtype,
) -> list[StabilityResult]:
    """
    Run direct-form and SOS filtering for one pole-radius case.

    Parameters
    ----------
    case:
        Pole-radius experiment definition.
    pole_angle_rad:
        Conjugate-pole angle in radians.
    section_count:
        Number of repeated second-order sections.
    num_samples:
        Impulse-response length in samples.
    dtype:
        Floating-point arithmetic dtype.

    Returns
    -------
    list[StabilityResult]
        Direct-form and SOS results.
    """
    sos = build_repeated_resonator_sos(
        pole_radius=case.pole_radius,
        pole_angle_rad=pole_angle_rad,
        section_count=section_count,
        dtype=dtype,
    )

    b, a = sos2tf(sos)
    validated_dtype = np.dtype(dtype)
    b = np.asarray(b, dtype=validated_dtype)
    a = np.asarray(a, dtype=validated_dtype)

    impulse = generate_unit_impulse(num_samples, validated_dtype)

    with np.errstate(over="ignore", invalid="ignore"):
        direct_output = lfilter(b, a, impulse)
        sos_output = sosfilt(sos, impulse)

    results: list[StabilityResult] = []

    for implementation, output in (
        ("Direct form", direct_output),
        ("SOS", sos_output),
    ):
        finite_mask = np.isfinite(output)
        all_finite = bool(np.all(finite_mask))

        finite_values = np.abs(output[finite_mask])
        peak_magnitude = (
            float(np.max(finite_values))
            if finite_values.size > 0
            else float("inf")
        )

        final_magnitude = (
            float(abs(output[-1]))
            if np.isfinite(output[-1])
            else float("inf")
        )

        results.append(
            StabilityResult(
                case_label=case.label,
                implementation=implementation,
                dtype_name=validated_dtype.name,
                pole_radius=case.pole_radius,
                output=np.asarray(output),
                peak_magnitude=peak_magnitude,
                final_magnitude=final_magnitude,
                all_finite=all_finite,
            )
        )

    return results
