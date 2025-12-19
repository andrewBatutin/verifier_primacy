"""Tests for LogprobsExplorer with mock backend."""

from collections.abc import Iterator

import numpy as np
import pytest

from verifier_primacy.logprobs.explorer import LogprobsExplorer
from verifier_primacy.logprobs.models import CompletionResult


class MockMLXBackend:
    """Mock MLX backend for testing without real models."""

    def __init__(self, vocab_size: int = 1000, model_family: str = "mock"):
        self._vocab_size = vocab_size
        self._model_family = model_family
        # Simple vocab: token_id -> character
        self._vocab = {i: chr(32 + (i % 95)) for i in range(vocab_size)}
        self._eos_token_id = 0

    def get_model_family(self) -> str:
        """Return the mock model family."""
        return self._model_family

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def encode(self, text: str) -> list[int]:
        return [ord(c) - 32 for c in text if 0 <= ord(c) - 32 < self._vocab_size]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._vocab.get(tid, "") for tid in tokens)

    def decode_token(self, token_id: int) -> str:
        return self._vocab.get(token_id, f"<{token_id}>")

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        # Return deterministic logits based on input
        logits = np.random.default_rng(len(token_ids)).standard_normal(self._vocab_size)
        return logits.astype(np.float32)

    def get_logprobs(self, token_ids: list[int]) -> np.ndarray:
        logits = self.get_logits(token_ids)
        # Softmax and log
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return np.log(probs + 1e-10)

    def generate_with_logprobs(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 5,
    ) -> Iterator[tuple[int, np.ndarray, list[tuple[int, float]]]]:
        """Generate deterministic tokens for testing."""
        tokens = list(prompt_tokens)

        for i in range(min(max_tokens, 5)):  # Generate max 5 tokens for tests
            logprobs = self.get_logprobs(tokens)

            # Get top-k
            top_k_indices = np.argsort(logprobs)[-top_k:][::-1]
            top_k_list = [(int(idx), float(logprobs[idx])) for idx in top_k_indices]

            # Pick the most likely token
            next_token = top_k_indices[0]

            yield next_token, logprobs, top_k_list

            tokens.append(next_token)

            # Stop if EOS
            if next_token == self._eos_token_id:
                break

    def score_tokens(
        self,
        prompt_tokens: list[int],
        continuation_tokens: list[int],
    ) -> list[float]:
        """Score continuation tokens."""
        tokens = list(prompt_tokens)
        scores = []

        for cont_token in continuation_tokens:
            logprobs = self.get_logprobs(tokens)
            scores.append(float(logprobs[cont_token]))
            tokens.append(cont_token)

        return scores


@pytest.fixture
def mock_backend() -> MockMLXBackend:
    """Provide a mock backend."""
    return MockMLXBackend(vocab_size=1000)


@pytest.fixture
def explorer(mock_backend: MockMLXBackend) -> LogprobsExplorer:
    """Provide an explorer with mock backend."""
    return LogprobsExplorer(backend=mock_backend)


class TestLogprobsExplorer:
    """Tests for LogprobsExplorer."""

    def test_complete_returns_result(self, explorer: LogprobsExplorer):
        """Complete should return a CompletionResult."""
        result = explorer.complete("Hello", max_tokens=3)
        assert isinstance(result, CompletionResult)
        assert result.prompt == "Hello"
        assert len(result.tokens) > 0

    def test_complete_has_tokens(self, explorer: LogprobsExplorer):
        """Result should have token information."""
        result = explorer.complete("Test", max_tokens=3)
        for token in result.tokens:
            assert token.chosen is not None
            assert token.chosen.token is not None
            assert token.chosen.logprob <= 0  # Log probs are <= 0
            assert 0 <= token.chosen.prob <= 1

    def test_complete_has_alternatives(self, explorer: LogprobsExplorer):
        """Result should include alternatives."""
        result = explorer.complete("Test", max_tokens=3, top_k=5)
        for token in result.tokens:
            # At least some alternatives (excluding chosen)
            assert len(token.alternatives) > 0

    def test_complete_respects_max_tokens(self, explorer: LogprobsExplorer):
        """Should not exceed max_tokens."""
        result = explorer.complete("Test", max_tokens=2)
        assert len(result.tokens) <= 2

    def test_get_logprobs(self, explorer: LogprobsExplorer):
        """Should score existing text."""
        result = explorer.get_logprobs(prompt="Hello", continuation=" world")
        assert result.text == " world"
        assert len(result.tokens) > 0
        assert result.total_logprob < 0  # Sum of negative values
        assert result.perplexity >= 1.0

    def test_compare_continuations(self, explorer: LogprobsExplorer):
        """Should rank continuations."""
        result = explorer.compare_continuations(
            prompt="Hello",
            continuations=[" world", " there", " friend"],
        )
        assert len(result.continuations) == 3
        assert len(result.ranking) == 3
        assert result.best is not None

    def test_compare_ranking_ordered(self, explorer: LogprobsExplorer):
        """Rankings should be valid indices."""
        result = explorer.compare_continuations(
            prompt="Test",
            continuations=["a", "b", "c"],
        )
        # All indices present
        assert set(result.ranking) == {0, 1, 2}

    def test_analyze_uncertainty(self, explorer: LogprobsExplorer):
        """Should find low-confidence tokens."""
        result = explorer.complete("Test", max_tokens=5)
        uncertain = explorer.analyze_uncertainty(result, threshold=0.99)
        # With random logits, some tokens will be uncertain
        # Just check it returns a list
        assert isinstance(uncertain, list)


class TestLogprobsExplorerIntegration:
    """Integration-style tests for LogprobsExplorer."""

    def test_complete_json_export(self, explorer: LogprobsExplorer):
        """Result should export to valid JSON."""
        result = explorer.complete("Hello", max_tokens=2)
        json_str = result.to_json()
        # Should be valid JSON
        import json

        data = json.loads(json_str)
        assert "prompt" in data
        assert "completion" in data
        assert "tokens" in data

    def test_complete_openai_format(self, explorer: LogprobsExplorer):
        """Result should export to OpenAI format."""
        result = explorer.complete("Hello", max_tokens=2)
        openai = result.to_openai_format()
        assert "content" in openai
        assert len(openai["content"]) > 0

    def test_perplexity_calculation(self, explorer: LogprobsExplorer):
        """Perplexity should be calculated correctly."""
        result = explorer.complete("Test", max_tokens=3)
        # Perplexity = exp(-avg_logprob)
        # With negative logprobs, perplexity > 1
        assert result.perplexity >= 1.0
