"""Tests for confidence scoring functions."""

import numpy as np
import pytest

from verifier_primacy.core.confidence import (
    calibrated_confidence,
    entropy_confidence,
    field_confidence,
    top_k_gap,
)


class TestEntropyConfidence:
    """Tests for entropy_confidence function."""

    def test_uniform_distribution_low_confidence(self, uniform_logits):
        """Uniform distribution should have low confidence."""
        conf = entropy_confidence(uniform_logits)
        assert conf < 0.1, f"Expected low confidence for uniform, got {conf}"

    def test_peaked_distribution_high_confidence(self, peaked_logits):
        """Peaked distribution should have high confidence."""
        conf = entropy_confidence(peaked_logits)
        assert conf > 0.9, f"Expected high confidence for peaked, got {conf}"

    def test_confidence_in_valid_range(self, uniform_logits, peaked_logits):
        """Confidence should always be in [0, 1]."""
        for logits in [uniform_logits, peaked_logits]:
            conf = entropy_confidence(logits)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} outside [0, 1]"

    def test_temperature_scaling(self, mixed_logits):
        """Higher temperature should reduce confidence."""
        conf_low_temp = entropy_confidence(mixed_logits, temperature=0.5)
        conf_high_temp = entropy_confidence(mixed_logits, temperature=2.0)
        assert conf_low_temp > conf_high_temp, "Higher temp should reduce confidence"


class TestTopKGap:
    """Tests for top_k_gap function."""

    def test_large_gap_high_confidence(self, peaked_logits):
        """Large gap between top-1 and top-2 should give high confidence."""
        conf = top_k_gap(peaked_logits, k=2)
        assert conf > 0.9, f"Expected high confidence for peaked, got {conf}"

    def test_small_gap_low_confidence(self, mixed_logits):
        """Small gap between top-1 and top-2 should give low confidence."""
        conf = top_k_gap(mixed_logits, k=2)
        assert conf < 0.5, f"Expected low confidence for close top-2, got {conf}"

    def test_invalid_k_raises(self, peaked_logits):
        """k < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            top_k_gap(peaked_logits, k=1)

    def test_k_larger_than_vocab_raises(self):
        """k larger than vocab size should raise ValueError."""
        small_logits = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 10 elements"):
            top_k_gap(small_logits, k=10)


class TestCalibratedConfidence:
    """Tests for calibrated_confidence function."""

    def test_identity_calibration(self, peaked_logits):
        """Scale=1.0, shift=0.0 should give same as entropy_confidence."""
        raw = entropy_confidence(peaked_logits)
        calibrated = calibrated_confidence(peaked_logits, 1.0, 0.0)
        assert abs(raw - calibrated) < 1e-6

    def test_scaling_increases_confidence(self, mixed_logits):
        """Scale > 1 should increase confidence."""
        raw = entropy_confidence(mixed_logits)
        calibrated = calibrated_confidence(mixed_logits, 1.5, 0.0)
        assert calibrated >= raw

    def test_output_clamped_to_valid_range(self, peaked_logits):
        """Output should be clamped to [0, 1] even with extreme calibration."""
        # Extreme scale that would push above 1.0
        conf = calibrated_confidence(peaked_logits, 10.0, 0.5)
        assert 0.0 <= conf <= 1.0


class TestFieldConfidence:
    """Tests for field_confidence aggregation."""

    def test_min_aggregation(self, peaked_logits, uniform_logits):
        """Min aggregation should return lowest confidence."""
        field_logits = [peaked_logits, uniform_logits]
        conf = field_confidence(field_logits, aggregation="min")
        expected = entropy_confidence(uniform_logits)
        assert abs(conf - expected) < 0.1

    def test_mean_aggregation(self, peaked_logits, uniform_logits):
        """Mean aggregation should return average confidence."""
        field_logits = [peaked_logits, uniform_logits]
        conf = field_confidence(field_logits, aggregation="mean")
        high = entropy_confidence(peaked_logits)
        low = entropy_confidence(uniform_logits)
        expected = (high + low) / 2
        assert abs(conf - expected) < 0.1

    def test_empty_field_returns_zero(self):
        """Empty field should return 0.0 confidence."""
        conf = field_confidence([])
        assert conf == 0.0

    def test_invalid_aggregation_raises(self, peaked_logits):
        """Invalid aggregation method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            field_confidence([peaked_logits], aggregation="invalid")
