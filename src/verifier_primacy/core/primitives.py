"""Verification primitives for structured output validation.

This module provides composable verification checks that can be applied
to extracted fields. Each check returns True if valid, False otherwise.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal


@dataclass
class FieldSpec:
    """Specification for a field in structured output.

    Args:
        name: Field name as it appears in the schema
        type: Expected type - "string", "integer", "number", "boolean", "null"
        enum: Optional list of allowed values for string fields
        min_value: Minimum value for numeric fields
        max_value: Maximum value for numeric fields
        pattern: Regex pattern for string fields
        required: Whether the field is required (default True)
        description: Human-readable description for the LLM
    """

    name: str
    type: Literal["string", "integer", "number", "boolean", "null"]
    enum: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    required: bool = True
    description: str = ""
    checks: list[Callable[[Any], bool]] = field(default_factory=list)


def check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches expected type.

    Args:
        value: Value to check
        expected_type: One of "string", "integer", "number", "boolean", "null"

    Returns:
        True if type matches, False otherwise
    """
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }

    expected = type_map.get(expected_type)
    if expected is None:
        raise ValueError(f"Unknown type: {expected_type}")

    # Special case: booleans are also integers in Python
    if expected_type == "integer" and isinstance(value, bool):
        return False

    return isinstance(value, expected)


def check_range(
    value: Any,
    min_value: float | None = None,
    max_value: float | None = None,
) -> bool:
    """Check if numeric value is within range.

    Args:
        value: Numeric value to check
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        True if value is within range, False otherwise
    """
    if not isinstance(value, (int, float)):
        return False

    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False

    return True


def check_not_null(value: Any) -> bool:
    """Check that value is not None or empty.

    Args:
        value: Value to check

    Returns:
        True if value is not null/empty, False otherwise
    """
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def check_regex(value: Any, pattern: str) -> bool:
    """Check if string value matches regex pattern.

    Args:
        value: String value to check
        pattern: Regex pattern to match

    Returns:
        True if pattern matches, False otherwise
    """
    if not isinstance(value, str):
        return False

    try:
        return bool(re.match(pattern, value))
    except re.error:
        return False


def check_enum(value: Any, allowed: list[str]) -> bool:
    """Check if value is one of the allowed values.

    Args:
        value: Value to check
        allowed: List of allowed values

    Returns:
        True if value is in allowed list, False otherwise
    """
    return value in allowed


def check_fuzzy_match(
    value: Any,
    known_values: list[str],
    threshold: float = 0.8,
) -> bool:
    """Check if string value fuzzy-matches any known value.

    Uses SequenceMatcher for similarity comparison. Useful for
    catching minor typos or variations in extracted values.

    Args:
        value: String value to check
        known_values: List of valid values to match against
        threshold: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        True if any known value matches above threshold, False otherwise
    """
    if not isinstance(value, str):
        return False

    for known in known_values:
        ratio = SequenceMatcher(None, value.lower(), known.lower()).ratio()
        if ratio >= threshold:
            return True

    return False


def check_cross_field(
    data: dict[str, Any],
    condition_field: str,
    condition_value: Any,
    target_field: str,
    target_check: Callable[[Any], bool],
) -> bool:
    """Apply conditional validation based on another field's value.

    Example: If currency is "USD", amount must be < 50000

    Args:
        data: Dictionary containing all extracted fields
        condition_field: Field name to check condition against
        condition_value: Value that triggers the check
        target_field: Field to validate if condition is met
        target_check: Validation function for target field

    Returns:
        True if condition not met OR target check passes, False otherwise
    """
    # If condition field doesn't match, check passes (not applicable)
    if data.get(condition_field) != condition_value:
        return True

    # Condition matches, apply target check
    target_value = data.get(target_field)
    return target_check(target_value)


@dataclass
class ValidationResult:
    """Result of validating a field or set of fields.

    Attributes:
        valid: Whether all checks passed
        field_name: Name of the field (if single field validation)
        errors: List of error messages for failed checks
        value: The validated value
    """

    valid: bool
    field_name: str | None = None
    errors: list[str] = field(default_factory=list)
    value: Any = None


def validate_field(value: Any, spec: FieldSpec) -> ValidationResult:
    """Validate a single field against its specification.

    Args:
        value: Extracted value to validate
        spec: FieldSpec defining validation rules

    Returns:
        ValidationResult with pass/fail status and any errors
    """
    errors = []

    # Check required
    if spec.required and not check_not_null(value):
        errors.append(f"Field '{spec.name}' is required but was empty/null")
        return ValidationResult(valid=False, field_name=spec.name, errors=errors, value=value)

    # Skip further validation if value is null and not required
    if value is None:
        return ValidationResult(valid=True, field_name=spec.name, value=value)

    # Check type
    if not check_type(value, spec.type):
        errors.append(f"Field '{spec.name}' expected type {spec.type}, got {type(value).__name__}")

    # Check enum
    if spec.enum is not None and not check_enum(value, spec.enum):
        errors.append(f"Field '{spec.name}' must be one of {spec.enum}, got '{value}'")

    # Check range
    if spec.type in ("integer", "number"):
        if not check_range(value, spec.min_value, spec.max_value):
            errors.append(
                f"Field '{spec.name}' must be in range [{spec.min_value}, {spec.max_value}]"
            )

    # Check pattern
    if spec.pattern is not None and not check_regex(value, spec.pattern):
        errors.append(f"Field '{spec.name}' must match pattern {spec.pattern}")

    # Run custom checks
    for i, check in enumerate(spec.checks):
        if not check(value):
            errors.append(f"Field '{spec.name}' failed custom check {i + 1}")

    return ValidationResult(
        valid=len(errors) == 0,
        field_name=spec.name,
        errors=errors,
        value=value,
    )
