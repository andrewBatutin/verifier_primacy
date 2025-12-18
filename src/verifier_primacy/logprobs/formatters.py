"""Formatters for pretty-printing logprobs results.

This module provides various formatting options for displaying
logprob information in human-readable formats.
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
    spans = []
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
