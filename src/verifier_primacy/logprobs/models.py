"""Pydantic models for logprobs data structures.

These models provide type-safe, serializable representations of token
log-probabilities with human-readable string representations and
easy JSON export for downstream processing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field


class TokenLogprob(BaseModel):
    """Log-probability information for a single token.

    Attributes:
        token: The decoded token text (e.g., " Paris", "hello").
        token_id: The token's ID in the vocabulary.
        logprob: The log-probability (natural log).
        prob: The probability (exp(logprob)), computed automatically.

    Example:
        >>> t = TokenLogprob(token=" Paris", token_id=12345, logprob=-0.165)
        >>> print(t)
        ' Paris' (84.7%)
        >>> t.prob
        0.8479...
    """

    token: str = Field(description="Decoded token text")
    token_id: int = Field(description="Token ID in vocabulary")
    logprob: float = Field(description="Log-probability (natural log)")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prob(self) -> float:
        """Probability computed from logprob."""
        return math.exp(self.logprob)

    def __str__(self) -> str:
        """Human-readable format: 'token' (XX.X%)."""
        return f"'{self.token}' ({self.prob * 100:.1f}%)"

    def __repr__(self) -> str:
        return (
            f"TokenLogprob(token={self.token!r}, logprob={self.logprob:.3f}, prob={self.prob:.3f})"
        )


class TokenWithAlternatives(BaseModel):
    """A generated token with its top-k alternatives.

    Shows what token was chosen and what alternatives the model considered,
    along with their respective probabilities.

    Attributes:
        chosen: The token that was actually sampled/selected.
        alternatives: Top-k alternative tokens the model considered.
        position: Position in the generated sequence (0-indexed).
        cumulative_logprob: Sum of logprobs up to and including this token.

    Example:
        >>> print(token_with_alts)
        [0] ' Paris' (84.7%) | alts: ' London' (8.9%), ' Berlin' (4.1%)
    """

    chosen: TokenLogprob = Field(description="The token that was sampled")
    alternatives: list[TokenLogprob] = Field(
        default_factory=list, description="Top-k alternative tokens"
    )
    position: int = Field(description="Position in sequence (0-indexed)")
    cumulative_logprob: float = Field(default=0.0, description="Sum of logprobs up to this point")

    def __str__(self) -> str:
        """Human-readable format showing chosen token and alternatives."""
        alts_str = ", ".join(str(a) for a in self.alternatives[:3])
        if alts_str:
            return f"[{self.position}] {self.chosen} | alts: {alts_str}"
        return f"[{self.position}] {self.chosen}"

    def __repr__(self) -> str:
        return (
            f"TokenWithAlternatives(chosen={self.chosen.token!r}, "
            f"alts={len(self.alternatives)}, pos={self.position})"
        )


class CompletionResult(BaseModel):
    """Complete result of a generation with logprobs.

    This is the main output from LogprobsExplorer.complete(). It contains
    the full generation result with per-token logprob information.

    Attributes:
        prompt: The input prompt text.
        completion: The generated completion text.
        tokens: List of tokens with their logprobs and alternatives.
        total_logprob: Sum of all token logprobs.
        perplexity: Perplexity of the generation (exp(-avg_logprob)).

    Example:
        >>> result = explorer.complete("The capital of France is")
        >>> print(result)
        Prompt: 'The capital of France is'
        Completion: ' Paris'
        ...
        >>> result.save_json("output.json")
    """

    prompt: str = Field(description="Input prompt text")
    completion: str = Field(description="Generated completion text")
    tokens: list[TokenWithAlternatives] = Field(
        default_factory=list, description="Per-token logprob information"
    )
    total_logprob: float = Field(default=0.0, description="Sum of all token logprobs")
    perplexity: float = Field(default=1.0, description="Perplexity of generation")

    def __str__(self) -> str:
        """Human-readable multi-line format."""
        lines = [
            f"Prompt: {self.prompt!r}",
            f"Completion: {self.completion!r}",
            "",
            "Token-by-token breakdown:",
        ]
        for t in self.tokens:
            lines.append(f"  {t}")
        lines.append("")
        lines.append(f"Total logprob: {self.total_logprob:.3f}")
        lines.append(f"Perplexity: {self.perplexity:.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CompletionResult(completion={self.completion!r}, "
            f"tokens={len(self.tokens)}, perplexity={self.perplexity:.2f})"
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Export to JSON string.

        Args:
            indent: JSON indentation level. None for compact output.

        Returns:
            JSON string representation.
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary for programmatic processing.

        Returns:
            Dictionary representation of the result.
        """
        return self.model_dump()

    def save_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Save to JSON file.

        Args:
            path: File path to save to.
            indent: JSON indentation level.
        """
        Path(path).write_text(self.to_json(indent=indent))

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible logprobs format.

        Returns a structure matching OpenAI's chat completion logprobs response.

        Returns:
            Dictionary in OpenAI logprobs format.
        """
        content = []
        for token_info in self.tokens:
            top_logprobs = [
                {
                    "token": alt.token,
                    "logprob": alt.logprob,
                    "bytes": list(alt.token.encode("utf-8")),
                }
                for alt in token_info.alternatives
            ]
            # Include chosen token in top_logprobs if not already there
            chosen_in_alts = any(
                alt.token_id == token_info.chosen.token_id for alt in token_info.alternatives
            )
            if not chosen_in_alts:
                top_logprobs.insert(
                    0,
                    {
                        "token": token_info.chosen.token,
                        "logprob": token_info.chosen.logprob,
                        "bytes": list(token_info.chosen.token.encode("utf-8")),
                    },
                )

            content.append(
                {
                    "token": token_info.chosen.token,
                    "logprob": token_info.chosen.logprob,
                    "bytes": list(token_info.chosen.token.encode("utf-8")),
                    "top_logprobs": top_logprobs,
                }
            )

        return {"content": content}


class TokenLogprobs(BaseModel):
    """Logprobs for a sequence of tokens (no generation, just scoring).

    Used when scoring existing text rather than generating new text.

    Attributes:
        text: The scored text.
        tokens: List of token logprobs.
        total_logprob: Sum of all token logprobs.
        perplexity: Perplexity of the text.
    """

    text: str = Field(description="The scored text")
    tokens: list[TokenLogprob] = Field(default_factory=list, description="Per-token logprobs")
    total_logprob: float = Field(default=0.0, description="Sum of token logprobs")
    perplexity: float = Field(default=1.0, description="Perplexity")

    def __str__(self) -> str:
        """Human-readable format."""
        token_strs = " ".join(f"[{t.token}:{t.prob:.2f}]" for t in self.tokens)
        return f"'{self.text}' -> {token_strs} (ppl: {self.perplexity:.2f})"

    def to_json(self, indent: int | None = 2) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return self.model_dump()

    def save_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Save to JSON file."""
        Path(path).write_text(self.to_json(indent=indent))


class ComparisonResult(BaseModel):
    """Result of comparing multiple continuations.

    Used when comparing which continuation is most likely given a prompt.

    Attributes:
        prompt: The input prompt.
        continuations: Scored continuations with their logprobs.
        ranking: Indices sorted by probability (most likely first).
    """

    prompt: str = Field(description="Input prompt")
    continuations: list[TokenLogprobs] = Field(
        default_factory=list, description="Scored continuations"
    )
    ranking: list[int] = Field(default_factory=list, description="Indices sorted by probability")

    @property
    def best(self) -> TokenLogprobs:
        """Get the most likely continuation."""
        if not self.ranking:
            raise ValueError("No continuations to rank")
        return self.continuations[self.ranking[0]]

    def __str__(self) -> str:
        """Human-readable ranking format."""
        lines = [f"Prompt: {self.prompt!r}", "", "Rankings:"]
        for i, idx in enumerate(self.ranking):
            cont = self.continuations[idx]
            lines.append(
                f"  {i + 1}. '{cont.text}' (logprob: {cont.total_logprob:.3f}, ppl: {cont.perplexity:.2f})"
            )
        return "\n".join(lines)

    def to_json(self, indent: int | None = 2) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return self.model_dump()

    def save_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Save to JSON file."""
        Path(path).write_text(self.to_json(indent=indent))
