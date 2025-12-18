"""MLX backend for model loading and generation with logprobs.

This module provides the MLX-specific implementation for loading models
and generating text with log-probability information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers compatible with the MLX backend."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Get end-of-sequence token ID."""
        ...


class MLXBackend:
    """MLX backend for model inference with logprobs.

    Handles model loading, tokenization, and generation with
    log-probability extraction on Apple Silicon.

    Example:
        >>> backend = MLXBackend.from_pretrained("mlx-community/Llama-3.2-1B-4bit")
        >>> for token_id, logprobs in backend.generate_with_logprobs(prompt_tokens):
        ...     print(f"Token: {token_id}, top logprob: {logprobs[token_id]:.3f}")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Initialize the MLX backend.

        Args:
            model: The loaded MLX model.
            tokenizer: The tokenizer for the model.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._vocab_size: int | None = None

    @classmethod
    def from_pretrained(cls, model_path: str) -> MLXBackend:
        """Load a model from HuggingFace Hub or local path.

        Args:
            model_path: HuggingFace model ID or local path.
                Examples:
                - "mlx-community/Llama-3.2-1B-4bit"
                - "/path/to/local/model"

        Returns:
            Initialized MLXBackend instance.

        Raises:
            ImportError: If mlx_lm is not installed.
        """
        try:
            import mlx_lm
        except ImportError as e:
            raise ImportError(
                "mlx_lm is required for MLX backend. Install with: pip install mlx-lm"
            ) from e

        model, tokenizer = mlx_lm.load(model_path)
        return cls(model=model, tokenizer=tokenizer)

    @property
    def model(self) -> Any:
        """Get the underlying MLX model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._vocab_size is None:
            # Try to get vocab size from tokenizer
            if hasattr(self._tokenizer, "vocab_size"):
                self._vocab_size = self._tokenizer.vocab_size
            elif hasattr(self._tokenizer, "__len__"):
                self._vocab_size = len(self._tokenizer)
            else:
                # Fallback: try to infer from model config
                config = getattr(self._model, "config", None)
                if config and hasattr(config, "vocab_size"):
                    self._vocab_size = config.vocab_size
                else:
                    raise ValueError("Could not determine vocabulary size")
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        """Get end-of-sequence token ID."""
        if hasattr(self._tokenizer, "eos_token_id"):
            return self._tokenizer.eos_token_id
        return None

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.
        """
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return self._tokenizer.decode(tokens)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text.

        Args:
            token_id: Single token ID.

        Returns:
            Decoded token text.
        """
        return self._tokenizer.decode([token_id])

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        """Get logits for the next token given input tokens.

        Args:
            token_ids: Input token IDs.

        Returns:
            Logits array of shape (vocab_size,).
        """
        try:
            import mlx.core as mx
        except ImportError as e:
            raise ImportError(
                "mlx is required for MLX backend. Install with: pip install mlx"
            ) from e

        # Convert to MLX array
        tokens = mx.array([token_ids])

        # Get model output
        logits = self._model(tokens)

        # Get last token logits
        last_logits = logits[0, -1, :]

        # Force evaluation and convert to numpy
        mx.eval(last_logits)
        return np.array(last_logits)

    def get_logprobs(self, token_ids: list[int]) -> np.ndarray:
        """Get log-probabilities for the next token.

        Args:
            token_ids: Input token IDs.

        Returns:
            Log-probability array of shape (vocab_size,).
        """
        try:
            import mlx.core as mx
        except ImportError as e:
            raise ImportError(
                "mlx is required for MLX backend. Install with: pip install mlx"
            ) from e

        logits = self.get_logits(token_ids)
        logits_mx = mx.array(logits)
        logprobs = mx.log_softmax(logits_mx)
        mx.eval(logprobs)
        return np.array(logprobs)

    def get_top_k_logprobs(
        self,
        token_ids: list[int],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Get top-k tokens with their log-probabilities.

        Args:
            token_ids: Input token IDs.
            k: Number of top tokens to return.

        Returns:
            List of (token_id, logprob) tuples sorted by probability.
        """
        logprobs = self.get_logprobs(token_ids)
        top_k_indices = np.argsort(logprobs)[-k:][::-1]
        return [(int(idx), float(logprobs[idx])) for idx in top_k_indices]

    def generate_with_logprobs(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 5,
    ) -> Iterator[tuple[int, np.ndarray, list[tuple[int, float]]]]:
        """Generate tokens with log-probabilities.

        Yields token-by-token with full logprob information for analysis.

        Args:
            prompt_tokens: Input token IDs (the prompt).
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Higher = more random.
            top_k: Number of top alternatives to include.

        Yields:
            Tuples of (token_id, full_logprobs, top_k_list) where:
            - token_id: The sampled token ID
            - full_logprobs: Full logprob array (vocab_size,)
            - top_k_list: List of (token_id, logprob) for top-k tokens
        """
        try:
            import mlx.core as mx
        except ImportError as e:
            raise ImportError(
                "mlx is required for MLX backend. Install with: pip install mlx"
            ) from e

        tokens = list(prompt_tokens)
        eos_id = self.eos_token_id

        for _ in range(max_tokens):
            # Get logits for next token
            logits = self.get_logits(tokens)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Compute log-probabilities
            logits_mx = mx.array(logits)
            logprobs_mx = mx.log_softmax(logits_mx)
            mx.eval(logprobs_mx)
            logprobs = np.array(logprobs_mx)

            # Get top-k
            top_k_indices = np.argsort(logprobs)[-top_k:][::-1]
            top_k_list = [(int(idx), float(logprobs[idx])) for idx in top_k_indices]

            # Sample next token
            probs_mx = mx.softmax(logits_mx)
            mx.eval(probs_mx)
            probs = np.array(probs_mx)

            # Multinomial sampling
            next_token = int(np.random.choice(len(probs), p=probs))

            yield next_token, logprobs, top_k_list

            # Update context
            tokens.append(next_token)

            # Check for EOS
            if eos_id is not None and next_token == eos_id:
                break

    def score_tokens(
        self,
        prompt_tokens: list[int],
        continuation_tokens: list[int],
    ) -> list[float]:
        """Score a sequence of tokens given a prompt.

        Computes the log-probability of each continuation token
        given the prompt and preceding continuation tokens.

        Args:
            prompt_tokens: The prompt token IDs.
            continuation_tokens: The continuation token IDs to score.

        Returns:
            List of log-probabilities, one per continuation token.
        """
        logprobs_list: list[float] = []
        tokens = list(prompt_tokens)

        for cont_token in continuation_tokens:
            # Get logprobs for next position
            logprobs = self.get_logprobs(tokens)
            logprobs_list.append(float(logprobs[cont_token]))
            tokens.append(cont_token)

        return logprobs_list


def list_available_models() -> list[str]:
    """List popular MLX models for quick start.

    Returns:
        List of model IDs that work well with MLX.
    """
    return [
        "mlx-community/Llama-3.2-1B-4bit",
        "mlx-community/Llama-3.2-3B-4bit",
        "mlx-community/Mistral-7B-v0.3-4bit",
        "mlx-community/gemma-2-2b-it-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "mlx-community/Qwen3-4B-4bit"
    ]
