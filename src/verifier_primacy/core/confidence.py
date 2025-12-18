"""Confidence scoring based on logit distributions.

This module provides functions to compute confidence scores from logit
distributions. All scores are in the range [0, 1] where 1 means high confidence.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[np.floating]


def entropy_confidence(logits: "Array", temperature: float = 1.0) -> float:
    """Compute confidence based on entropy of the logit distribution.

    Low entropy (peaked distribution) → high confidence
    High entropy (uniform distribution) → low confidence

    Args:
        logits: Raw logits from the model, shape (vocab_size,)
        temperature: Temperature for softmax, default 1.0

    Returns:
        Confidence score in [0, 1]

    Example:
        >>> logits = np.array([10.0, 0.0, 0.0, 0.0])  # Peaked
        >>> entropy_confidence(logits)
        0.98  # High confidence
    """
    # Apply temperature and compute probabilities
    scaled = logits / temperature
    # Numerical stability: subtract max
    scaled = scaled - np.max(scaled)
    exp_logits = np.exp(scaled)
    probs = exp_logits / np.sum(exp_logits)

    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(logits))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Convert to confidence: low entropy = high confidence
    confidence = 1.0 - normalized_entropy

    return float(np.clip(confidence, 0.0, 1.0))


def top_k_gap(logits: "Array", k: int = 2) -> float:
    """Compute confidence based on gap between top-1 and top-k probability.

    Large gap → model is certain about top choice → high confidence
    Small gap → model is uncertain between top choices → low confidence

    Args:
        logits: Raw logits from the model, shape (vocab_size,)
        k: Compare top-1 with top-k (default: 2, i.e., top-2)

    Returns:
        Confidence score in [0, 1]

    Example:
        >>> logits = np.array([10.0, 9.9, 0.0, 0.0])  # Close top-2
        >>> top_k_gap(logits, k=2)
        0.05  # Low confidence (uncertain between top 2)
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(logits) < k:
        raise ValueError(f"logits must have at least {k} elements")

    # Compute probabilities
    scaled = logits - np.max(logits)
    exp_logits = np.exp(scaled)
    probs = exp_logits / np.sum(exp_logits)

    # Get top-k probabilities
    top_k_probs = np.sort(probs)[-k:]
    top_1_prob = top_k_probs[-1]
    top_k_prob = top_k_probs[0]  # k-th highest

    # Gap is difference between top-1 and top-k
    gap = top_1_prob - top_k_prob

    # Normalize: max possible gap is ~1.0 (when top-1 has all probability)
    return float(np.clip(gap, 0.0, 1.0))


def calibrated_confidence(
    logits: "Array",
    calibration_scale: float = 1.0,
    calibration_shift: float = 0.0,
) -> float:
    """Compute calibrated confidence score.

    Applies a learned calibration transform to the raw confidence score
    to better align predicted confidence with actual accuracy.

    Args:
        logits: Raw logits from the model, shape (vocab_size,)
        calibration_scale: Multiplicative calibration factor
        calibration_shift: Additive calibration factor

    Returns:
        Calibrated confidence score in [0, 1]

    Note:
        Calibration parameters should be learned from a held-out dataset
        by comparing predicted confidence to actual accuracy.
    """
    raw_confidence = entropy_confidence(logits)

    # Apply Platt scaling (logistic calibration)
    calibrated = calibration_scale * raw_confidence + calibration_shift

    # Ensure output is in [0, 1]
    return float(np.clip(calibrated, 0.0, 1.0))


def field_confidence(
    field_logits: list["Array"],
    aggregation: str = "min",
) -> float:
    """Aggregate confidence scores for a multi-token field.

    Args:
        field_logits: List of logits for each token in the field
        aggregation: How to aggregate - "min", "mean", or "product"

    Returns:
        Aggregated confidence score in [0, 1]
    """
    if not field_logits:
        return 0.0

    confidences = [entropy_confidence(logits) for logits in field_logits]

    if aggregation == "min":
        return float(min(confidences))
    elif aggregation == "mean":
        return float(np.mean(confidences))
    elif aggregation == "product":
        return float(np.prod(confidences))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
