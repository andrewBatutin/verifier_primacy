"""Decision boundary detection for logprobs analysis.

Identifies positions where the model is uncertain between different
response types (e.g., tool call vs text generation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifier_primacy.logprobs.models import CompletionResult, TokenLogprob


# Token patterns for classification
TOOL_CALL_TOKENS = frozenset(
    [
        "<tool_call>",
        "<function",
        "```",
        '{"',
        "{\n",
    ]
)

TEXT_START_TOKENS = frozenset(
    [
        "I",
        " I",
        "Sure",
        " Sure",
        "Let",
        " Let",
        "Here",
        " Here",
        "The",
        " The",
        "Yes",
        " Yes",
        "No",
        " No",
    ]
)


@dataclass
class DecisionBoundary:
    """A position where the model shows significant uncertainty."""

    position: int
    chosen: TokenLogprob
    alternative: TokenLogprob
    boundary_type: str
    risk_level: str  # "low", "medium", "high"


def classify_boundary_type(chosen_token: str, alt_token: str) -> str:
    """Classify the type of decision boundary.

    Args:
        chosen_token: The token that was selected.
        alt_token: The alternative token with significant probability.

    Returns:
        Boundary type string.
    """
    chosen_is_tool = any(t in chosen_token for t in TOOL_CALL_TOKENS)
    alt_is_tool = any(t in alt_token for t in TOOL_CALL_TOKENS)
    chosen_is_text = any(t in chosen_token for t in TEXT_START_TOKENS)
    alt_is_text = any(t in alt_token for t in TEXT_START_TOKENS)

    if chosen_is_tool and alt_is_text:
        return "tool_vs_text"
    if chosen_is_text and alt_is_tool:
        return "text_vs_tool"
    if chosen_token.strip().lower() in ("yes", "no") or alt_token.strip().lower() in ("yes", "no"):
        return "yes_vs_no"

    return "semantic_split"


def assess_risk_level(boundary_type: str, alt_prob: float) -> str:
    """Assess the risk level of a decision boundary.

    Args:
        boundary_type: The type of boundary.
        alt_prob: Probability of the alternative path.

    Returns:
        Risk level: "low", "medium", or "high".
    """
    # Tool vs text boundaries are always higher risk
    if boundary_type in ("tool_vs_text", "text_vs_tool"):
        if alt_prob > 0.3:
            return "high"
        if alt_prob > 0.15:
            return "medium"
        return "low"

    # Other boundaries
    if alt_prob > 0.4:
        return "high"
    if alt_prob > 0.2:
        return "medium"
    return "low"


def find_decision_boundaries(
    result: CompletionResult,
    threshold: float = 0.1,
) -> list[DecisionBoundary]:
    """Find positions where the model shows significant uncertainty.

    Args:
        result: The completion result to analyze.
        threshold: Minimum probability for an alternative to be considered.

    Returns:
        List of decision boundaries found.
    """
    boundaries: list[DecisionBoundary] = []

    for token in result.tokens:
        for alt in token.alternatives:
            if alt.prob >= threshold:
                boundary_type = classify_boundary_type(
                    token.chosen.token,
                    alt.token,
                )
                risk_level = assess_risk_level(boundary_type, alt.prob)

                boundaries.append(
                    DecisionBoundary(
                        position=token.position,
                        chosen=token.chosen,
                        alternative=alt,
                        boundary_type=boundary_type,
                        risk_level=risk_level,
                    )
                )
                # Only take first significant alternative per position
                break

    return boundaries


def filter_critical_boundaries(
    boundaries: list[DecisionBoundary],
) -> list[DecisionBoundary]:
    """Filter to only critical boundaries (tool vs text, high risk).

    Args:
        boundaries: All detected boundaries.

    Returns:
        Only critical boundaries worth exploring.
    """
    return [
        b
        for b in boundaries
        if b.boundary_type in ("tool_vs_text", "text_vs_tool") or b.risk_level == "high"
    ]
