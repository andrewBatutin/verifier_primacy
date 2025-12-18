"""Core functionality for verifier primacy."""

from verifier_primacy.core.confidence import (
    calibrated_confidence,
    entropy_confidence,
    top_k_gap,
)
from verifier_primacy.core.primitives import FieldSpec
from verifier_primacy.core.routing import Router, RoutingDecision
from verifier_primacy.core.vocab import VocabAnalyzer

__all__ = [
    "VocabAnalyzer",
    "entropy_confidence",
    "top_k_gap",
    "calibrated_confidence",
    "FieldSpec",
    "Router",
    "RoutingDecision",
]
