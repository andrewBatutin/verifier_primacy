"""Routing logic for human review decisions.

This module provides the Router class for making pass/review/reject decisions
based on confidence scores and validation results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from verifier_primacy.core.primitives import FieldSpec, ValidationResult, validate_field


class RoutingDecision(Enum):
    """Possible routing decisions for an extraction result."""

    PASS = auto()  # High confidence, all validations pass → accept automatically
    REVIEW = auto()  # Medium confidence or some validations fail → human review
    REJECT = auto()  # Low confidence or critical validations fail → reject


@dataclass
class RoutingResult:
    """Result of a routing decision.

    Attributes:
        decision: The routing decision (PASS, REVIEW, REJECT)
        confidence: Overall confidence score [0, 1]
        flagged_fields: Fields that triggered review/rejection
        validation_errors: List of validation error messages
        metadata: Additional context about the decision
    """

    decision: RoutingDecision
    confidence: float
    flagged_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingThresholds:
    """Thresholds for routing decisions.

    Attributes:
        pass_threshold: Minimum confidence for automatic pass
        reject_threshold: Maximum confidence for automatic reject
        require_all_valid: If True, all validations must pass for PASS
    """

    pass_threshold: float = 0.9
    reject_threshold: float = 0.5
    require_all_valid: bool = True

    def __post_init__(self) -> None:
        if self.reject_threshold >= self.pass_threshold:
            raise ValueError("reject_threshold must be < pass_threshold")


class Router:
    """Routes extraction results to pass/review/reject based on confidence.

    The Router makes decisions based on:
    1. Per-field confidence scores
    2. Validation results from primitives
    3. Configurable thresholds

    Example:
        >>> router = Router(thresholds=RoutingThresholds(pass_threshold=0.9))
        >>> result = router.decide(extracted_data, confidence_scores, schema)
        >>> if result.decision == RoutingDecision.REVIEW:
        ...     send_to_human(extracted_data, result.flagged_fields)
    """

    def __init__(
        self,
        thresholds: RoutingThresholds | None = None,
        custom_rules: list[Callable[[dict[str, Any], dict[str, float]], RoutingDecision | None]]
        | None = None,
    ) -> None:
        """Initialize the Router.

        Args:
            thresholds: Confidence thresholds for decisions
            custom_rules: Additional rules that can override default logic.
                         Each rule takes (data, confidence) and returns
                         a decision or None to defer to next rule.
        """
        self.thresholds = thresholds or RoutingThresholds()
        self.custom_rules = custom_rules or []

    def decide(
        self,
        data: dict[str, Any],
        confidence: dict[str, float],
        schema: list[FieldSpec] | None = None,
    ) -> RoutingResult:
        """Make a routing decision for extracted data.

        Args:
            data: Extracted field values
            confidence: Per-field confidence scores
            schema: Optional schema for validation

        Returns:
            RoutingResult with decision and supporting information
        """
        # Apply custom rules first
        for rule in self.custom_rules:
            custom_decision = rule(data, confidence)
            if custom_decision is not None:
                return RoutingResult(
                    decision=custom_decision,
                    confidence=self._aggregate_confidence(confidence),
                    metadata={"triggered_by": "custom_rule"},
                )

        # Collect flagged fields and validation errors
        flagged_fields: list[str] = []
        validation_errors: list[str] = []

        # Check confidence thresholds
        for field_name, conf in confidence.items():
            if conf < self.thresholds.reject_threshold:
                flagged_fields.append(field_name)
            elif conf < self.thresholds.pass_threshold:
                flagged_fields.append(field_name)

        # Run validation if schema provided
        if schema:
            for spec in schema:
                value = data.get(spec.name)
                result = validate_field(value, spec)
                if not result.valid:
                    flagged_fields.append(spec.name)
                    validation_errors.extend(result.errors)

        # Calculate aggregate confidence
        overall_confidence = self._aggregate_confidence(confidence)

        # Make decision
        decision = self._make_decision(
            overall_confidence, flagged_fields, validation_errors
        )

        return RoutingResult(
            decision=decision,
            confidence=overall_confidence,
            flagged_fields=list(set(flagged_fields)),
            validation_errors=validation_errors,
        )

    def _aggregate_confidence(self, confidence: dict[str, float]) -> float:
        """Aggregate per-field confidence into overall score.

        Uses minimum confidence as the aggregate - a chain is only
        as strong as its weakest link.
        """
        if not confidence:
            return 0.0
        return min(confidence.values())

    def _make_decision(
        self,
        confidence: float,
        flagged_fields: list[str],
        validation_errors: list[str],
    ) -> RoutingDecision:
        """Determine the routing decision based on all factors."""
        # Reject if confidence is very low
        if confidence < self.thresholds.reject_threshold:
            return RoutingDecision.REJECT

        # Reject if there are critical validation errors and require_all_valid
        if self.thresholds.require_all_valid and validation_errors:
            # Check if it's low enough to reject vs review
            if confidence < (self.thresholds.pass_threshold + self.thresholds.reject_threshold) / 2:
                return RoutingDecision.REJECT
            return RoutingDecision.REVIEW

        # Review if there are flagged fields
        if flagged_fields:
            return RoutingDecision.REVIEW

        # Review if confidence is below pass threshold
        if confidence < self.thresholds.pass_threshold:
            return RoutingDecision.REVIEW

        # All checks pass
        return RoutingDecision.PASS


def create_conditional_router(
    pass_condition: Callable[[dict[str, Any], dict[str, float]], bool],
    reject_condition: Callable[[dict[str, Any], dict[str, float]], bool] | None = None,
) -> Router:
    """Create a Router with custom pass/reject conditions.

    Args:
        pass_condition: Function that returns True if result should pass
        reject_condition: Function that returns True if result should reject

    Returns:
        Router configured with the custom conditions
    """

    def custom_rule(
        data: dict[str, Any], confidence: dict[str, float]
    ) -> RoutingDecision | None:
        if reject_condition and reject_condition(data, confidence):
            return RoutingDecision.REJECT
        if pass_condition(data, confidence):
            return RoutingDecision.PASS
        return None

    return Router(custom_rules=[custom_rule])
