"""Verifier Primacy - Constrained decoding via logit-level rule verification.

This library provides tools for verifying and constraining LLM outputs at the
logit level, ensuring structured outputs are valid before sampling.

Example:
    >>> from verifier_primacy import create_json_verifier, FieldSpec
    >>> schema = [FieldSpec(name="action", type="string", enum=["search", "create"])]
    >>> verifier = create_json_verifier(tokenizer, schema)
    >>> logits = verifier.apply(logits, state)  # Mask invalid tokens
"""

# Logprobs module - lazy import to avoid MLX dependency if not needed
from verifier_primacy import logprobs
from verifier_primacy.core.confidence import (
    calibrated_confidence,
    entropy_confidence,
    top_k_gap,
)
from verifier_primacy.core.primitives import (
    FieldSpec,
    check_cross_field,
    check_fuzzy_match,
    check_not_null,
    check_range,
    check_regex,
    check_type,
)
from verifier_primacy.core.routing import Router, RoutingDecision
from verifier_primacy.core.vocab import VocabAnalyzer
from verifier_primacy.rules import VerificationRule

__version__ = "0.1.0"

__all__ = [
    # Core
    "VocabAnalyzer",
    "entropy_confidence",
    "top_k_gap",
    "calibrated_confidence",
    # Primitives
    "FieldSpec",
    "check_type",
    "check_range",
    "check_not_null",
    "check_regex",
    "check_fuzzy_match",
    "check_cross_field",
    # Routing
    "Router",
    "RoutingDecision",
    # Rules
    "VerificationRule",
    # Logprobs
    "logprobs",
]
