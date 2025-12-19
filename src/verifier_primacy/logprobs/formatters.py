"""Formatters for pretty-printing logprobs results.

This module provides various formatting options for displaying
logprob information in human-readable formats, with focus on
showing the VALUE that logprobs provide to users.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verifier_primacy.logprobs.models import (
        ComparisonResult,
        CompletionResult,
        TokenWithAlternatives,
    )

# Confidence thresholds for interpretation
HIGH_CONFIDENCE_THRESHOLD = 0.7  # >= 70%
LOW_CONFIDENCE_THRESHOLD = 0.4  # < 40%

# Perplexity interpretation thresholds
EXCELLENT_PPL = 1.5  # < 1.5: Very natural
GOOD_PPL = 3.0  # 1.5-3.0: Natural
MODERATE_PPL = 6.0  # 3.0-6.0: Acceptable
# > 6.0: Model struggled


def _get_confidence_label(prob: float) -> tuple[str, str]:
    """Get confidence label and description.

    Args:
        prob: Probability value (0-1).

    Returns:
        Tuple of (short_label, description).
    """
    if prob >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH", "Model is confident"
    elif prob >= LOW_CONFIDENCE_THRESHOLD:
        return "MED", "Moderate confidence"
    else:
        return "LOW", "Model uncertain - verify this"


def _get_perplexity_interpretation(ppl: float) -> tuple[str, str]:
    """Interpret perplexity value.

    Args:
        ppl: Perplexity value.

    Returns:
        Tuple of (rating, description).
    """
    if ppl < EXCELLENT_PPL:
        return "EXCELLENT", "Very natural, model highly confident"
    elif ppl < GOOD_PPL:
        return "GOOD", "Natural output, model confident"
    elif ppl < MODERATE_PPL:
        return "MODERATE", "Acceptable, some uncertainty"
    else:
        return "HIGH", "Model struggled with this generation"


def print_logprobs(result: CompletionResult, max_alts: int = 3) -> None:
    """Log a completion result with detailed token information.

    Args:
        result: The completion result to display.
        max_alts: Maximum alternatives to show per token.
    """
    logger.info("Prompt: %r", result.prompt)
    logger.info("Completion: %r", result.completion)
    logger.info("=" * 60)
    logger.info("Token-by-token breakdown:")
    logger.info("-" * 60)

    for token in result.tokens:
        _log_token(token, max_alts)

    logger.info("-" * 60)
    logger.info("Total logprob: %.4f", result.total_logprob)
    logger.info("Perplexity: %.2f", result.perplexity)
    logger.info("=" * 60)


def _log_token(token: TokenWithAlternatives, max_alts: int) -> None:
    """Log a single token with alternatives."""
    chosen = token.chosen
    prob_bar = _make_prob_bar(chosen.prob, width=20)

    logger.info("[%3d] %r", token.position, chosen.token)
    logger.info("      prob: %6.2f%% %s", chosen.prob * 100, prob_bar)
    logger.info("      logprob: %.4f", chosen.logprob)

    if token.alternatives:
        logger.info("      alternatives:")
        for alt in token.alternatives[:max_alts]:
            alt_bar = _make_prob_bar(alt.prob, width=10)
            logger.info("        - %r: %6.2f%% %s", alt.token, alt.prob * 100, alt_bar)


def _make_prob_bar(prob: float, width: int = 20) -> str:
    """Create an ASCII probability bar."""
    filled = int(prob * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_compact(result: CompletionResult) -> str:
    """Format result in a compact single-line-per-token format.

    Args:
        result: The completion result to format.

    Returns:
        Compact string representation.
    """
    lines = [f"Prompt: {result.prompt!r}", f"Completion: {result.completion!r}", ""]

    for token in result.tokens:
        alts = ", ".join(f"{a.token!r}:{a.prob:.1%}" for a in token.alternatives[:2])
        line = f"[{token.position}] {token.chosen.token!r} ({token.chosen.prob:.1%})"
        if alts:
            line += f" | {alts}"
        lines.append(line)

    lines.append(f"\nPerplexity: {result.perplexity:.2f}")
    return "\n".join(lines)


def format_markdown(result: CompletionResult) -> str:
    """Format result as Markdown for documentation or notebooks.

    Args:
        result: The completion result to format.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        "## Completion Result",
        "",
        f"**Prompt:** `{result.prompt}`",
        "",
        f"**Completion:** `{result.completion}`",
        "",
        "### Token Breakdown",
        "",
        "| Pos | Token | Prob | Logprob | Top Alternative |",
        "|-----|-------|------|---------|-----------------|",
    ]

    for token in result.tokens:
        top_alt = token.alternatives[0] if token.alternatives else None
        alt_str = f"`{top_alt.token}` ({top_alt.prob:.1%})" if top_alt else "-"
        lines.append(
            f"| {token.position} | `{token.chosen.token}` | "
            f"{token.chosen.prob:.1%} | {token.chosen.logprob:.3f} | {alt_str} |"
        )

    lines.extend(
        [
            "",
            "### Summary",
            "",
            f"- **Total logprob:** {result.total_logprob:.4f}",
            f"- **Perplexity:** {result.perplexity:.2f}",
            f"- **Token count:** {len(result.tokens)}",
        ]
    )

    return "\n".join(lines)


def format_comparison(result: ComparisonResult) -> str:
    """Format a comparison result.

    Args:
        result: The comparison result to format.

    Returns:
        Formatted string showing rankings.
    """
    lines = [
        f"Prompt: {result.prompt!r}",
        "",
        "Rankings (most likely first):",
        "-" * 40,
    ]

    for rank, idx in enumerate(result.ranking):
        cont = result.continuations[idx]
        marker = " <-- best" if rank == 0 else ""
        lines.append(
            f"{rank + 1}. {cont.text!r}: logprob={cont.total_logprob:.3f}, "
            f"ppl={cont.perplexity:.2f}{marker}"
        )

    return "\n".join(lines)


def confidence_heatmap(result: CompletionResult, width: int = 50) -> str:
    """Create an ASCII heatmap showing confidence per token.

    High confidence = dense characters, low confidence = sparse.

    Args:
        result: The completion result to visualize.
        width: Width of the heatmap.

    Returns:
        ASCII heatmap string.
    """
    if not result.tokens:
        return "No tokens to display"

    chars = " ░▒▓█"  # Low to high confidence

    lines = ["Confidence Heatmap:", ""]

    # Token labels line
    token_labels = ""
    heatmap_line = ""

    for token in result.tokens:
        prob = token.chosen.prob
        # Map probability to character (0-1 -> 0-4)
        char_idx = min(int(prob * 5), 4)
        char = chars[char_idx]

        # Use token text length for spacing
        token_text = token.chosen.token.replace("\n", "\\n")
        if len(token_text) > 8:
            token_text = token_text[:7] + "…"

        token_labels += f"{token_text:8}"
        heatmap_line += char * 8

        if len(heatmap_line) >= width:
            lines.append(token_labels)
            lines.append(heatmap_line)
            lines.append("")
            token_labels = ""
            heatmap_line = ""

    if heatmap_line:
        lines.append(token_labels)
        lines.append(heatmap_line)

    lines.extend(
        [
            "",
            "Legend: ' '=0-20%, '░'=20-40%, '▒'=40-60%, '▓'=60-80%, '█'=80-100%",
        ]
    )

    return "\n".join(lines)


def to_html(result: CompletionResult) -> str:
    """Generate HTML visualization for Jupyter notebooks.

    Args:
        result: The completion result to visualize.

    Returns:
        HTML string with colored tokens.
    """
    # Generate colored spans based on probability
    spans: list[str] = []
    for token in result.tokens:
        prob = token.chosen.prob
        # Color from red (low) to green (high)
        if prob < 0.3:
            color = "#ff6b6b"  # Red
        elif prob < 0.6:
            color = "#ffd93d"  # Yellow
        else:
            color = "#6bcb77"  # Green

        escaped_token = (
            token.chosen.token.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "↵")
        )

        # Create tooltip with details
        tooltip = f"prob: {prob:.2%}, logprob: {token.chosen.logprob:.3f}"
        if token.alternatives:
            top_alt = token.alternatives[0]
            tooltip += f", alt: {top_alt.token!r} ({top_alt.prob:.1%})"

        spans.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'margin: 1px; border-radius: 3px; display: inline-block;" '
            f'title="{tooltip}">{escaped_token}</span>'
        )

    html = f"""
    <div style="font-family: monospace; line-height: 1.8;">
        <p><strong>Prompt:</strong> {result.prompt}</p>
        <p><strong>Completion:</strong></p>
        <div style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
            {"".join(spans)}
        </div>
        <p style="margin-top: 10px;">
            <strong>Perplexity:</strong> {result.perplexity:.2f} |
            <strong>Total logprob:</strong> {result.total_logprob:.4f}
        </p>
        <p style="font-size: 0.9em; color: #666;">
            Colors: <span style="background: #ff6b6b; padding: 2px 6px;">low (&lt;30%)</span>
            <span style="background: #ffd93d; padding: 2px 6px;">medium (30-60%)</span>
            <span style="background: #6bcb77; padding: 2px 6px;">high (&gt;60%)</span>
        </p>
    </div>
    """
    return html


# =============================================================================
# VALUE-FOCUSED FORMATTERS
# These formatters help users understand WHAT VALUE logprobs provide
# =============================================================================


def format_confidence_report(result: CompletionResult) -> str:
    """Format a confidence report showing where the model is certain vs uncertain.

    This is the key value proposition: know when to trust the output.

    Args:
        result: The completion result to analyze.

    Returns:
        Human-readable confidence report.

    Example output:
        === CONFIDENCE REPORT ===
        Completion: " Paris is the capital"

        Overall: HIGH CONFIDENCE (85%)

        Token-by-Token Analysis:
        [HIGH]  " Paris" (92%) - Model is confident
        [HIGH]  " is" (88%) - Model is confident
        [MED]   " the" (55%) - Moderate confidence
        [HIGH]  " capital" (95%) - Model is confident

        Risky Tokens: 0 tokens below 40% confidence
        Action: Output appears reliable
    """
    if not result.tokens:
        return "No tokens to analyze"

    lines = [
        "=== CONFIDENCE REPORT ===",
        f'Completion: "{result.completion}"',
        "",
    ]

    # Calculate average confidence
    avg_confidence = sum(t.chosen.prob for t in result.tokens) / len(result.tokens)
    overall_label, _ = _get_confidence_label(avg_confidence)
    lines.append(f"Overall: {overall_label} CONFIDENCE ({avg_confidence:.0%})")
    lines.append("")
    lines.append("Token-by-Token Analysis:")

    risky_tokens: list[tuple[int, str]] = []

    for token in result.tokens:
        prob = token.chosen.prob
        label, desc = _get_confidence_label(prob)

        # Format token for display
        token_text = repr(token.chosen.token)
        if len(token_text) > 20:
            token_text = token_text[:17] + "..."

        line = f"  [{label:4}] {token_text:22} ({prob:5.0%}) - {desc}"

        # Add alternative info for uncertain tokens
        if prob < LOW_CONFIDENCE_THRESHOLD and token.alternatives:
            top_alt = token.alternatives[0]
            line += f", considered {repr(top_alt.token)} ({top_alt.prob:.0%})"
            risky_tokens.append((token.position, token.chosen.token))

        lines.append(line)

    lines.append("")
    lines.append(f"Risky Tokens: {len(risky_tokens)} token(s) below 40% confidence")

    if risky_tokens:
        lines.append("  Positions: " + ", ".join(str(pos) for pos, _tok in risky_tokens))
        lines.append("Action: VERIFY these tokens before trusting output")
    else:
        lines.append("Action: Output appears reliable")

    return "\n".join(lines)


def format_alternatives_insight(result: CompletionResult, min_alt_prob: float = 0.05) -> str:
    """Show what the model almost said - the roads not taken.

    This helps understand model reasoning and catch potential errors.

    Args:
        result: The completion result to analyze.
        min_alt_prob: Minimum probability for alternatives to show (default 5%).

    Returns:
        Human-readable alternatives analysis.

    Example output:
        === ALTERNATIVE PATHS ===
        What the model almost said...

        Position 0: Chose " Paris" (92%)
          Also considered:
          - " London" (4%) - close alternative
          - " Berlin" (2%)

        Position 2: Chose " the" (55%)
          Also considered:
          - " a" (30%) - significant alternative!
          - " France" (8%)

        Key Insight: Position 2 had a close call - " a" was almost chosen
    """
    if not result.tokens:
        return "No tokens to analyze"

    lines = [
        "=== ALTERNATIVE PATHS ===",
        "What the model almost said...",
        "",
    ]

    significant_alternatives: list[tuple[int, str, str, float]] = []

    for token in result.tokens:
        if not token.alternatives:
            continue

        # Filter alternatives above threshold
        relevant_alts = [a for a in token.alternatives if a.prob >= min_alt_prob]
        if not relevant_alts:
            continue

        chosen_text = repr(token.chosen.token)
        lines.append(f"Position {token.position}: Chose {chosen_text} ({token.chosen.prob:.0%})")
        lines.append("  Also considered:")

        for alt in relevant_alts[:3]:  # Show top 3 alternatives
            alt_text = repr(alt.token)
            note = ""

            # Flag significant alternatives (>20% of chosen probability)
            if alt.prob > token.chosen.prob * 0.3:
                note = " - significant alternative!"
                significant_alternatives.append(
                    (token.position, token.chosen.token, alt.token, alt.prob)
                )

            lines.append(f"    - {alt_text} ({alt.prob:.0%}){note}")

        lines.append("")

    if significant_alternatives:
        lines.append("Key Insights:")
        for pos, chosen, alt, alt_prob in significant_alternatives:
            lines.append(
                f"  - Position {pos}: {repr(alt)} ({alt_prob:.0%}) was a strong alternative to {repr(chosen)}"
            )
    else:
        lines.append("Key Insight: Model was decisive, no close alternatives")

    return "\n".join(lines)


def format_quality_summary(result: CompletionResult) -> str:
    """One-glance quality assessment of the generation.

    Provides actionable summary for quick decision making.

    Args:
        result: The completion result to summarize.

    Returns:
        Concise quality summary.

    Example output:
        === QUALITY SUMMARY ===
        Prompt: "The capital of France is"
        Output: " Paris"

        Perplexity: 1.18 (EXCELLENT - Very natural, model highly confident)
        Avg Confidence: 85% (HIGH)
        Token Count: 1
        Uncertain Tokens: 0 of 1 (0%)

        Verdict: HIGH QUALITY - Output is reliable
    """
    if not result.tokens:
        return "No tokens to analyze"

    # Calculate metrics
    avg_confidence = sum(t.chosen.prob for t in result.tokens) / len(result.tokens)
    uncertain_count = sum(1 for t in result.tokens if t.chosen.prob < LOW_CONFIDENCE_THRESHOLD)
    uncertain_pct = uncertain_count / len(result.tokens)

    ppl_rating, ppl_desc = _get_perplexity_interpretation(result.perplexity)
    conf_label, _ = _get_confidence_label(avg_confidence)

    lines = [
        "=== QUALITY SUMMARY ===",
        f'Prompt: "{result.prompt}"',
        f'Output: "{result.completion}"',
        "",
        f"Perplexity: {result.perplexity:.2f} ({ppl_rating} - {ppl_desc})",
        f"Avg Confidence: {avg_confidence:.0%} ({conf_label})",
        f"Token Count: {len(result.tokens)}",
        f"Uncertain Tokens: {uncertain_count} of {len(result.tokens)} ({uncertain_pct:.0%})",
        "",
    ]

    # Determine overall verdict
    if result.perplexity < GOOD_PPL and avg_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        verdict = "HIGH QUALITY - Output is reliable"
    elif result.perplexity < MODERATE_PPL and avg_confidence >= LOW_CONFIDENCE_THRESHOLD:
        verdict = "ACCEPTABLE - Output likely correct, spot-check recommended"
    else:
        verdict = "REVIEW NEEDED - Model showed uncertainty, verify output"

    lines.append(f"Verdict: {verdict}")

    return "\n".join(lines)


def log_value_report(result: CompletionResult) -> None:
    """Log a complete value-focused report.

    Combines all value formatters into one comprehensive output.

    Args:
        result: The completion result to report on.
    """
    logger.info("\n%s", format_quality_summary(result))
    logger.info("\n%s", format_confidence_report(result))
    logger.info("\n%s", format_alternatives_insight(result))


# =============================================================================
# DECISION BOUNDARY FORMATTERS
# =============================================================================


def _classify_token_type(token: str) -> str:
    """Classify a token as tool_call, text_start, or other."""
    from verifier_primacy.logprobs.boundaries import TEXT_START_TOKENS, TOOL_CALL_TOKENS

    if any(t in token for t in TOOL_CALL_TOKENS):
        return "tool_call"
    if any(t in token for t in TEXT_START_TOKENS):
        return "text_start"
    return "other"


def format_position_zero_analysis(token: TokenWithAlternatives) -> str:
    """Format analysis of position 0 (first token) alternatives.

    Always shows what the model considered at the critical first-token
    decision point, regardless of probability threshold.

    Args:
        token: The first token with alternatives.

    Returns:
        Formatted position 0 analysis.
    """
    lines = [
        "=== POSITION 0 ANALYSIS ===",
        "First token decision point (always shown)",
        "",
    ]

    chosen = token.chosen
    chosen_type = _classify_token_type(chosen.token)
    type_label = (
        " (tool)"
        if chosen_type == "tool_call"
        else " (text)"
        if chosen_type == "text_start"
        else ""
    )

    lines.append(f"  Chosen: {repr(chosen.token)} ({chosen.prob:.0%}){type_label}")
    lines.append("")

    if token.alternatives:
        lines.append("  Alternatives:")
        for i, alt in enumerate(token.alternatives[:5], 1):
            alt_type = _classify_token_type(alt.token)
            alt_label = (
                " - tool call"
                if alt_type == "tool_call"
                else " - text"
                if alt_type == "text_start"
                else ""
            )
            lines.append(f"    {i}. {repr(alt.token)} ({alt.prob:.0%}){alt_label}")
        lines.append("")

        # Assessment
        tool_alts = [a for a in token.alternatives if _classify_token_type(a.token) == "tool_call"]
        text_alts = [a for a in token.alternatives if _classify_token_type(a.token) == "text_start"]

        if chosen_type == "tool_call":
            if text_alts:
                total_text_prob = sum(a.prob for a in text_alts)
                lines.append(
                    f"  Assessment: Model chose tool call. Text alternatives had {total_text_prob:.0%} combined probability."
                )
            else:
                lines.append(
                    "  Assessment: Model confidently chose tool call with no text alternatives."
                )
        elif chosen_type == "text_start":
            if tool_alts:
                total_tool_prob = sum(a.prob for a in tool_alts)
                lines.append(
                    f"  Assessment: Model chose text response. Tool call had {total_tool_prob:.0%} probability."
                )
            else:
                lines.append(
                    "  Assessment: Model chose text response with no tool call alternatives."
                )
        else:
            lines.append("  Assessment: First token is neither typical tool call nor text start.")
    else:
        lines.append("  No alternatives recorded for position 0.")

    return "\n".join(lines)


def format_boundary_report(
    boundaries: list,  # list[DecisionBoundary] - avoid circular import
    primary_result: CompletionResult,
    alternative_results: dict,  # position -> CompletionResult
) -> str:
    """Format a decision boundary analysis report.

    Shows where the model was uncertain between different response types
    and what each path leads to.

    Args:
        boundaries: List of detected decision boundaries.
        primary_result: The original completion result.
        alternative_results: Map of position to alternative completion.

    Returns:
        Human-readable boundary report.
    """
    if not boundaries:
        return "=== NO DECISION BOUNDARIES DETECTED ===\nModel was decisive throughout generation."

    lines = [
        "=== DECISION BOUNDARY ANALYSIS ===",
        f"Found {len(boundaries)} critical decision point(s)",
        "",
    ]

    for boundary in boundaries:
        pos = boundary.position
        chosen = boundary.chosen
        alt = boundary.alternative

        # Determine risk indicator
        risk_emoji = {"high": "!!!", "medium": "!!", "low": "!"}
        risk_indicator = risk_emoji.get(boundary.risk_level, "")

        lines.extend(
            [
                f"--- Position {pos}: {boundary.boundary_type.upper()} {risk_indicator} ---",
                "",
                f"  Path A ({chosen.prob:.0%}): {repr(chosen.token)}",
            ]
        )

        # Show primary path completion
        lines.append(f"    -> {repr(primary_result.completion)}")

        # Show alternative path if generated
        if pos in alternative_results:
            alt_result = alternative_results[pos]
            lines.extend(
                [
                    "",
                    f"  Path B ({alt.prob:.0%}): {repr(alt.token)}",
                    f"    -> {repr(alt_result.completion)}",
                ]
            )

        # Risk assessment
        lines.extend(
            [
                "",
                f"  Risk Level: {boundary.risk_level.upper()}",
            ]
        )

        if boundary.boundary_type == "tool_vs_text":
            lines.append("  Warning: Model could have called a tool OR generated text")
        elif boundary.boundary_type == "text_vs_tool":
            lines.append("  Warning: Model generated text but could have called a tool")

        lines.append("")

    # Summary
    high_risk = sum(1 for b in boundaries if b.risk_level == "high")
    tool_boundaries = sum(1 for b in boundaries if "tool" in b.boundary_type)

    lines.extend(
        [
            "=== SUMMARY ===",
            f"High risk boundaries: {high_risk}",
            f"Tool/text divergences: {tool_boundaries}",
            "",
        ]
    )

    if high_risk > 0:
        lines.append("Recommendation: Consider constrained decoding or lower temperature")
    elif tool_boundaries > 0:
        lines.append("Recommendation: Review tool-calling logic for this prompt type")
    else:
        lines.append("Recommendation: Model behavior is acceptable but has variance")

    return "\n".join(lines)
