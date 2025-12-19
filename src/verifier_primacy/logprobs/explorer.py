"""LogprobsExplorer - Main interface for exploring model log-probabilities.

This module provides the primary user-facing API for generating text
with log-probability information from local MLX models.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from verifier_primacy.logprobs.models import (
    ComparisonResult,
    CompletionResult,
    TokenLogprob,
    TokenLogprobs,
    TokenWithAlternatives,
)

if TYPE_CHECKING:
    from verifier_primacy.backends.mlx_backend import MLXBackend


def apply_template(prompt: str, model_family: str, disable_thinking: bool = True) -> str:
    """Apply model-specific prompt template.

    Args:
        prompt: The input prompt.
        model_family: Model family string (e.g., "qwen3", "llama").
        disable_thinking: Whether to disable thinking mode for Qwen3 models.

    Returns:
        Templated prompt string.
    """
    if model_family == "qwen3" and disable_thinking:
        return prompt + "<think>\n\n</think>\n\n"
    return prompt


class LogprobsExplorer:
    """Explore model outputs with log-probability information.

    The main interface for generating text and analyzing log-probabilities
    from local MLX models. Designed for macOS/iOS engineers who want to
    understand what their models are "thinking".

    Example:
        >>> from verifier_primacy.logprobs import LogprobsExplorer
        >>>
        >>> # Load a model
        >>> explorer = LogprobsExplorer.from_pretrained("mlx-community/Llama-3.2-1B-4bit")
        >>>
        >>> # Generate with logprobs
        >>> result = explorer.complete("The capital of France is", top_k=5)
        >>> print(result)
        Prompt: 'The capital of France is'
        Completion: ' Paris'
        ...
        >>>
        >>> # Export to JSON
        >>> result.save_json("output.json")
    """

    def __init__(self, backend: MLXBackend) -> None:
        """Initialize the explorer with a backend.

        Args:
            backend: The MLX backend for model inference.
        """
        self._backend = backend

    @classmethod
    def from_pretrained(cls, model_path: str) -> LogprobsExplorer:
        """Load a model and create an explorer.

        This is the recommended way to create a LogprobsExplorer.

        Args:
            model_path: HuggingFace model ID or local path.
                Examples:
                - "mlx-community/Llama-3.2-1B-4bit"
                - "/path/to/local/model"

        Returns:
            Initialized LogprobsExplorer ready to use.

        Raises:
            ImportError: If MLX dependencies are not installed.

        Example:
            >>> explorer = LogprobsExplorer.from_pretrained(
            ...     "mlx-community/Llama-3.2-1B-4bit"
            ... )
        """
        from verifier_primacy.backends.mlx_backend import MLXBackend

        backend = MLXBackend.from_pretrained(model_path)
        return cls(backend=backend)

    @property
    def backend(self) -> MLXBackend:
        """Get the underlying backend."""
        return self._backend

    def complete(
        self,
        prompt: str,
        max_tokens: int = 50,
        top_k: int = 5,
        temperature: float = 1.0,
        disable_thinking: bool = True,
    ) -> CompletionResult:
        """Generate a completion with log-probability information.

        Args:
            prompt: The input text to complete.
            max_tokens: Maximum tokens to generate.
            top_k: Number of top alternatives to track per token.
            temperature: Sampling temperature (higher = more random).
            disable_thinking: Whether to disable thinking mode for Qwen3 models.
                Defaults to True for simpler output.

        Returns:
            CompletionResult with full logprob information.

        Example:
            >>> result = explorer.complete("Hello, ", max_tokens=10)
            >>> print(result.completion)
            'world!'
            >>> for token in result.tokens:
            ...     print(f"{token.chosen.token}: {token.chosen.prob:.3f}")
        """
        # Apply model-specific templating
        model_family = self._backend.get_model_family()
        templated_prompt = apply_template(prompt, model_family, disable_thinking)

        # Encode prompt
        prompt_tokens = self._backend.encode(templated_prompt)

        # Generate with logprobs
        generated_tokens: list[TokenWithAlternatives] = []
        cumulative_logprob = 0.0

        for token_id, logprobs, top_k_list in self._backend.generate_with_logprobs(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            # Get chosen token info
            chosen_logprob = float(logprobs[token_id])
            cumulative_logprob += chosen_logprob

            chosen = TokenLogprob(
                token=self._backend.decode_token(token_id),
                token_id=token_id,
                logprob=chosen_logprob,
            )

            # Get alternatives (excluding chosen if present)
            alternatives = []
            for alt_id, alt_logprob in top_k_list:
                if alt_id != token_id:
                    alternatives.append(
                        TokenLogprob(
                            token=self._backend.decode_token(alt_id),
                            token_id=alt_id,
                            logprob=alt_logprob,
                        )
                    )

            generated_tokens.append(
                TokenWithAlternatives(
                    chosen=chosen,
                    alternatives=alternatives[:top_k],
                    position=len(generated_tokens),
                    cumulative_logprob=cumulative_logprob,
                )
            )

        # Build completion text
        completion = "".join(t.chosen.token for t in generated_tokens)

        # Calculate perplexity
        if generated_tokens:
            avg_logprob = cumulative_logprob / len(generated_tokens)
            perplexity = math.exp(-avg_logprob)
        else:
            perplexity = 1.0

        return CompletionResult(
            prompt=prompt,
            completion=completion,
            tokens=generated_tokens,
            total_logprob=cumulative_logprob,
            perplexity=perplexity,
        )

    def get_logprobs(
        self,
        prompt: str,
        continuation: str,
    ) -> TokenLogprobs:
        """Get log-probabilities for existing text.

        Scores how likely the continuation is given the prompt,
        without generating new tokens.

        Args:
            prompt: The context/prompt text.
            continuation: The text to score.

        Returns:
            TokenLogprobs with per-token probability information.

        Example:
            >>> result = explorer.get_logprobs(
            ...     prompt="The capital of France is",
            ...     continuation=" Paris"
            ... )
            >>> print(f"Perplexity: {result.perplexity:.2f}")
        """
        prompt_tokens = self._backend.encode(prompt)
        continuation_tokens = self._backend.encode(continuation)

        # Score each continuation token
        logprobs_list = self._backend.score_tokens(prompt_tokens, continuation_tokens)

        tokens = []
        for token_id, logprob in zip(continuation_tokens, logprobs_list, strict=True):
            tokens.append(
                TokenLogprob(
                    token=self._backend.decode_token(token_id),
                    token_id=token_id,
                    logprob=logprob,
                )
            )

        total_logprob = sum(logprobs_list)
        if tokens:
            avg_logprob = total_logprob / len(tokens)
            perplexity = math.exp(-avg_logprob)
        else:
            perplexity = 1.0

        return TokenLogprobs(
            text=continuation,
            tokens=tokens,
            total_logprob=total_logprob,
            perplexity=perplexity,
        )

    def compare_continuations(
        self,
        prompt: str,
        continuations: list[str],
    ) -> ComparisonResult:
        """Compare multiple possible continuations.

        Scores each continuation and ranks them by likelihood.
        Useful for understanding which option the model prefers.

        Args:
            prompt: The context/prompt text.
            continuations: List of possible continuations to compare.

        Returns:
            ComparisonResult with rankings and scores.

        Example:
            >>> result = explorer.compare_continuations(
            ...     prompt="The capital of France is",
            ...     continuations=[" Paris", " London", " Berlin"]
            ... )
            >>> print(f"Best: {result.best.text}")
            Best:  Paris
        """
        scored: list[TokenLogprobs] = []

        for continuation in continuations:
            scored.append(self.get_logprobs(prompt, continuation))

        # Rank by total logprob (higher = more likely)
        ranking = sorted(
            range(len(scored)),
            key=lambda i: scored[i].total_logprob,
            reverse=True,
        )

        return ComparisonResult(
            prompt=prompt,
            continuations=scored,
            ranking=ranking,
        )

    def analyze_uncertainty(
        self,
        result: CompletionResult,
        threshold: float = 0.5,
    ) -> list[TokenWithAlternatives]:
        """Find tokens where the model was uncertain.

        Args:
            result: A completion result to analyze.
            threshold: Probability threshold below which to flag uncertainty.

        Returns:
            List of tokens where chosen probability < threshold.
        """
        uncertain = []
        for token in result.tokens:
            if token.chosen.prob < threshold:
                uncertain.append(token)
        return uncertain

    def complete_from_tokens(
        self,
        prompt: str,
        prefix_token_ids: list[int],
        max_tokens: int = 20,
        top_k: int = 5,
        temperature: float = 1.0,
        disable_thinking: bool = True,
    ) -> CompletionResult:
        """Generate completion starting from specific tokens.

        Used for exploring alternative paths by forcing a prefix.

        Args:
            prompt: The original prompt text (for result metadata).
            prefix_token_ids: Token IDs to use as prefix before generation.
            max_tokens: Maximum additional tokens to generate.
            top_k: Number of top alternatives to track per token.
            temperature: Sampling temperature.
            disable_thinking: Whether to disable thinking mode for Qwen3.

        Returns:
            CompletionResult starting from the forced prefix.
        """
        # Apply model-specific templating
        model_family = self._backend.get_model_family()
        templated_prompt = apply_template(prompt, model_family, disable_thinking)

        # Encode prompt and combine with prefix
        prompt_tokens = self._backend.encode(templated_prompt)
        context_tokens = prompt_tokens + prefix_token_ids

        # Generate with logprobs
        generated_tokens: list[TokenWithAlternatives] = []
        cumulative_logprob = 0.0

        # First, add the prefix tokens (without logprobs since they were forced)
        prefix_text = ""
        for i, token_id in enumerate(prefix_token_ids):
            token_text = self._backend.decode_token(token_id)
            prefix_text += token_text
            generated_tokens.append(
                TokenWithAlternatives(
                    chosen=TokenLogprob(
                        token=token_text,
                        token_id=token_id,
                        logprob=0.0,  # Forced, no logprob
                    ),
                    alternatives=[],
                    position=i,
                    cumulative_logprob=0.0,
                )
            )

        # Now generate additional tokens
        for token_id, logprobs, top_k_list in self._backend.generate_with_logprobs(
            prompt_tokens=context_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            chosen_logprob = float(logprobs[token_id])
            cumulative_logprob += chosen_logprob

            chosen = TokenLogprob(
                token=self._backend.decode_token(token_id),
                token_id=token_id,
                logprob=chosen_logprob,
            )

            alternatives = []
            for alt_id, alt_logprob in top_k_list:
                if alt_id != token_id:
                    alternatives.append(
                        TokenLogprob(
                            token=self._backend.decode_token(alt_id),
                            token_id=alt_id,
                            logprob=alt_logprob,
                        )
                    )

            generated_tokens.append(
                TokenWithAlternatives(
                    chosen=chosen,
                    alternatives=alternatives[:top_k],
                    position=len(generated_tokens),
                    cumulative_logprob=cumulative_logprob,
                )
            )

            context_tokens.append(token_id)

        # Build completion text
        completion = "".join(t.chosen.token for t in generated_tokens)

        # Calculate perplexity (only for generated, not forced tokens)
        gen_count = len(generated_tokens) - len(prefix_token_ids)
        if gen_count > 0:
            avg_logprob = cumulative_logprob / gen_count
            perplexity = math.exp(-avg_logprob)
        else:
            perplexity = 1.0

        return CompletionResult(
            prompt=prompt,
            completion=completion,
            tokens=generated_tokens,
            total_logprob=cumulative_logprob,
            perplexity=perplexity,
        )


def list_models() -> list[str]:
    """List popular MLX models for quick start.

    Returns:
        List of model IDs that work well with this library.
    """
    from verifier_primacy.backends.mlx_backend import list_available_models

    return list_available_models()
