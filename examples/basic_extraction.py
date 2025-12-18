"""Basic extraction example demonstrating verifier_primacy.

This example shows how to:
1. Define a schema for structured output
2. Create a verifier
3. Use confidence scoring
4. Route results based on confidence

Run with:
    uv run python examples/basic_extraction.py
"""

import numpy as np

# Simulated for demo - in real use, import from verifier_primacy
from verifier_primacy.core.confidence import entropy_confidence, top_k_gap
from verifier_primacy.core.primitives import FieldSpec, validate_field
from verifier_primacy.core.routing import Router, RoutingDecision, RoutingThresholds


def demo_confidence_scoring():
    """Demonstrate confidence scoring on different logit distributions."""
    print("=== Confidence Scoring Demo ===\n")

    # High confidence: peaked distribution
    peaked_logits = np.full(1000, -100.0, dtype=np.float32)
    peaked_logits[42] = 0.0

    print("Peaked distribution (model is certain):")
    print(f"  Entropy confidence: {entropy_confidence(peaked_logits):.3f}")
    print(f"  Top-k gap (k=2):    {top_k_gap(peaked_logits, k=2):.3f}")

    # Low confidence: uniform distribution
    uniform_logits = np.zeros(1000, dtype=np.float32)

    print("\nUniform distribution (model is uncertain):")
    print(f"  Entropy confidence: {entropy_confidence(uniform_logits):.3f}")
    print(f"  Top-k gap (k=2):    {top_k_gap(uniform_logits, k=2):.3f}")

    # Medium confidence: top-2 competition
    mixed_logits = np.full(1000, -100.0, dtype=np.float32)
    mixed_logits[42] = 0.0
    mixed_logits[43] = -0.1  # Close second choice

    print("\nClose top-2 (model is torn between options):")
    print(f"  Entropy confidence: {entropy_confidence(mixed_logits):.3f}")
    print(f"  Top-k gap (k=2):    {top_k_gap(mixed_logits, k=2):.3f}")


def demo_validation():
    """Demonstrate validation primitives."""
    print("\n=== Validation Primitives Demo ===\n")

    # Define schema
    schema = [
        FieldSpec(
            name="action",
            type="string",
            enum=["search", "create", "delete"],
        ),
        FieldSpec(
            name="count",
            type="integer",
            min_value=0,
            max_value=100,
        ),
    ]

    # Test cases
    test_cases = [
        {"action": "search", "count": 10},  # Valid
        {"action": "update", "count": 10},  # Invalid action
        {"action": "search", "count": 150},  # Count out of range
        {"action": "search", "count": "ten"},  # Wrong type
    ]

    for data in test_cases:
        print(f"Data: {data}")
        for spec in schema:
            result = validate_field(data.get(spec.name), spec)
            status = "✓" if result.valid else "✗"
            errors = f" - {', '.join(result.errors)}" if result.errors else ""
            print(f"  {status} {spec.name}{errors}")
        print()


def demo_routing():
    """Demonstrate routing decisions based on confidence."""
    print("=== Routing Demo ===\n")

    router = Router(
        thresholds=RoutingThresholds(
            pass_threshold=0.9,
            reject_threshold=0.5,
        )
    )

    # Define schema for validation
    schema = [
        FieldSpec(name="vendor", type="string"),
        FieldSpec(name="amount", type="number", min_value=0),
    ]

    # Scenario 1: High confidence, valid data → PASS
    print("Scenario 1: High confidence extraction")
    result = router.decide(
        data={"vendor": "Acme Corp", "amount": 1499.99},
        confidence={"vendor": 0.95, "amount": 0.92},
        schema=schema,
    )
    print(f"  Decision: {result.decision.name}")
    print(f"  Overall confidence: {result.confidence:.2f}")

    # Scenario 2: Medium confidence → REVIEW
    print("\nScenario 2: Medium confidence extraction")
    result = router.decide(
        data={"vendor": "Acme Corp", "amount": 1499.99},
        confidence={"vendor": 0.95, "amount": 0.75},  # Amount uncertain
        schema=schema,
    )
    print(f"  Decision: {result.decision.name}")
    print(f"  Flagged fields: {result.flagged_fields}")

    # Scenario 3: Low confidence → REJECT
    print("\nScenario 3: Low confidence extraction")
    result = router.decide(
        data={"vendor": "???", "amount": -100},
        confidence={"vendor": 0.3, "amount": 0.4},
        schema=schema,
    )
    print(f"  Decision: {result.decision.name}")
    print(f"  Validation errors: {result.validation_errors}")


def main():
    """Run all demos."""
    print("Verifier Primacy - Basic Examples")
    print("=" * 50)

    demo_confidence_scoring()
    demo_validation()
    demo_routing()

    print("\n" + "=" * 50)
    print("Done! See examples/ for more detailed demos.")


if __name__ == "__main__":
    main()
