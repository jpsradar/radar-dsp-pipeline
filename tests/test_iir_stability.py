"""
test_iir_stability.py

Unit tests for src.iir_stability.

Purpose
-------
Validate practical IIR stability and numerical robustness behavior used in
Stage 06.

Engineering contracts checked here:

- stable SOS impulse responses remain finite
- stable responses decay relative to their peak
- invalid pole radii are rejected
- generated impulse shape and dtype are preserved
"""

import numpy as np
import pytest

from src.iir_stability import (
    StabilityCase,
    build_repeated_resonator_sos,
    generate_unit_impulse,
    run_stability_case,
)


def test_stable_sos_impulse_response_remains_finite_and_decays() -> None:
    """
    Stable SOS filtering should remain finite and decay from its peak.

    A pole radius below one must produce a bounded impulse response for the
    tested finite observation interval.
    """
    results = run_stability_case(
        case=StabilityCase(
            label="Stable",
            pole_radius=0.90,
        ),
        pole_angle_rad=np.deg2rad(20.0),
        section_count=2,
        num_samples=256,
        dtype=np.dtype(np.float64),
    )

    sos_result = next(
        result for result in results
        if result.implementation == "SOS"
    )

    assert sos_result.all_finite
    assert np.all(np.isfinite(sos_result.output))
    assert sos_result.final_magnitude < sos_result.peak_magnitude


def test_build_repeated_resonator_sos_rejects_non_positive_radius() -> None:
    """
    Resonator construction should reject non-positive pole radii.

    Pole radius must be strictly positive before coefficients are generated.
    """
    with pytest.raises(
        ValueError,
        match="pole_radius must be positive",
    ):
        build_repeated_resonator_sos(
            pole_radius=0.0,
            pole_angle_rad=np.deg2rad(20.0),
            section_count=2,
            dtype=np.dtype(np.float64),
        )


def test_generate_unit_impulse_preserves_shape_and_dtype() -> None:
    """
    Unit impulse generation should preserve requested length and dtype.

    Stage 06 compares float32 and float64 behavior, so dtype preservation is
    part of the experiment contract.
    """
    impulse = generate_unit_impulse(
        num_samples=128,
        dtype=np.dtype(np.float32),
    )

    assert impulse.shape == (128,)
    assert impulse.dtype == np.float32
    assert impulse[0] == pytest.approx(1.0)
    assert np.count_nonzero(impulse[1:]) == 0
