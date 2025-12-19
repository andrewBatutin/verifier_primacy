"""Logprobs module - Tools for exploring LLM log-probabilities.

This module provides a simple, transparent API for exploring what
local MLX models are "thinking" through log-probability analysis.

Quick Start:
    >>> from verifier_primacy.logprobs import LogprobsExplorer
    >>>
    >>> # Load a model
    >>> explorer = LogprobsExplorer.from_pretrained("mlx-community/Llama-3.2-1B-4bit")
    >>>
    >>> # Generate with logprobs
    >>> result = explorer.complete("The capital of France is", top_k=5)
    >>> print(result)
    >>>
    >>> # Export to JSON
    >>> result.save_json("output.json")
    >>>
    >>> # Or get raw dict
    >>> data = result.to_dict()
"""

from verifier_primacy.logprobs.explorer import LogprobsExplorer, list_models
from verifier_primacy.logprobs.formatters import (
    confidence_heatmap,
    format_alternatives_insight,
    format_compact,
    format_comparison,
    format_confidence_report,
    format_markdown,
    format_quality_summary,
    log_value_report,
    print_logprobs,
    to_html,
)
from verifier_primacy.logprobs.inspector import TokenInfo, TokenInspection, TokenInspector
from verifier_primacy.logprobs.models import (
    ComparisonResult,
    CompletionResult,
    TokenLogprob,
    TokenLogprobs,
    TokenWithAlternatives,
)

__all__ = [
    # Main interface
    "LogprobsExplorer",
    "list_models",
    # Models
    "TokenLogprob",
    "TokenWithAlternatives",
    "CompletionResult",
    "TokenLogprobs",
    "ComparisonResult",
    # Inspector
    "TokenInspector",
    "TokenInspection",
    "TokenInfo",
    # Formatters
    "print_logprobs",
    "format_compact",
    "format_markdown",
    "format_comparison",
    "confidence_heatmap",
    "to_html",
    # Value-focused formatters
    "format_confidence_report",
    "format_alternatives_insight",
    "format_quality_summary",
    "log_value_report",
]
