"""Tests for logprobs Pydantic models."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from verifier_primacy.logprobs.models import (
    ComparisonResult,
    CompletionResult,
    TokenLogprob,
    TokenLogprobs,
    TokenWithAlternatives,
)


class TestTokenLogprob:
    """Tests for TokenLogprob model."""

    def test_prob_computed_from_logprob(self):
        """Probability should be exp(logprob)."""
        token = TokenLogprob(token="test", token_id=42, logprob=-1.0)
        assert abs(token.prob - math.exp(-1.0)) < 1e-6

    def test_high_logprob_high_prob(self):
        """Logprob close to 0 should give probability close to 1."""
        token = TokenLogprob(token="test", token_id=42, logprob=-0.01)
        assert token.prob > 0.99

    def test_low_logprob_low_prob(self):
        """Very negative logprob should give probability close to 0."""
        token = TokenLogprob(token="test", token_id=42, logprob=-10.0)
        assert token.prob < 0.001

    def test_str_format(self):
        """String representation should show token and percentage."""
        token = TokenLogprob(token=" Paris", token_id=12345, logprob=-0.165)
        result = str(token)
        assert "Paris" in result
        assert "%" in result

    def test_json_serialization(self):
        """Model should serialize to JSON."""
        token = TokenLogprob(token="hello", token_id=1, logprob=-0.5)
        json_str = token.model_dump_json()
        data = json.loads(json_str)
        assert data["token"] == "hello"
        assert data["token_id"] == 1
        assert abs(data["logprob"] - (-0.5)) < 1e-6

    def test_from_json(self):
        """Model should deserialize from JSON."""
        json_str = '{"token": "world", "token_id": 2, "logprob": -0.3}'
        token = TokenLogprob.model_validate_json(json_str)
        assert token.token == "world"
        assert token.token_id == 2


class TestTokenWithAlternatives:
    """Tests for TokenWithAlternatives model."""

    def test_basic_creation(self):
        """Should create with chosen token and alternatives."""
        chosen = TokenLogprob(token="Paris", token_id=1, logprob=-0.1)
        alts = [
            TokenLogprob(token="London", token_id=2, logprob=-1.0),
            TokenLogprob(token="Berlin", token_id=3, logprob=-2.0),
        ]
        token = TokenWithAlternatives(
            chosen=chosen, alternatives=alts, position=0, cumulative_logprob=-0.1
        )
        assert token.chosen.token == "Paris"
        assert len(token.alternatives) == 2
        assert token.position == 0

    def test_str_format(self):
        """String should show position, chosen, and alternatives."""
        chosen = TokenLogprob(token="A", token_id=1, logprob=-0.1)
        alts = [TokenLogprob(token="B", token_id=2, logprob=-1.0)]
        token = TokenWithAlternatives(
            chosen=chosen, alternatives=alts, position=5, cumulative_logprob=-0.5
        )
        result = str(token)
        assert "[5]" in result
        assert "A" in result

    def test_empty_alternatives(self):
        """Should work with no alternatives."""
        chosen = TokenLogprob(token="only", token_id=1, logprob=-0.5)
        token = TokenWithAlternatives(
            chosen=chosen, alternatives=[], position=0, cumulative_logprob=-0.5
        )
        assert len(token.alternatives) == 0


class TestCompletionResult:
    """Tests for CompletionResult model."""

    @pytest.fixture
    def sample_result(self) -> CompletionResult:
        """Create a sample completion result."""
        tokens = [
            TokenWithAlternatives(
                chosen=TokenLogprob(token=" Paris", token_id=1, logprob=-0.165),
                alternatives=[
                    TokenLogprob(token=" London", token_id=2, logprob=-2.41),
                ],
                position=0,
                cumulative_logprob=-0.165,
            ),
            TokenWithAlternatives(
                chosen=TokenLogprob(token=".", token_id=3, logprob=-0.02),
                alternatives=[],
                position=1,
                cumulative_logprob=-0.185,
            ),
        ]
        return CompletionResult(
            prompt="The capital of France is",
            completion=" Paris.",
            tokens=tokens,
            total_logprob=-0.185,
            perplexity=1.2,
        )

    def test_to_json(self, sample_result: CompletionResult):
        """Should export to valid JSON string."""
        json_str = sample_result.to_json()
        data = json.loads(json_str)
        assert data["prompt"] == "The capital of France is"
        assert data["completion"] == " Paris."
        assert len(data["tokens"]) == 2

    def test_to_dict(self, sample_result: CompletionResult):
        """Should export to dictionary."""
        data = sample_result.to_dict()
        assert isinstance(data, dict)
        assert data["prompt"] == "The capital of France is"
        assert data["tokens"][0]["chosen"]["token"] == " Paris"

    def test_save_and_load_json(self, sample_result: CompletionResult):
        """Should save to file and load back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sample_result.save_json(path)

            # Load back
            loaded = CompletionResult.model_validate_json(path.read_text())
            assert loaded.prompt == sample_result.prompt
            assert loaded.completion == sample_result.completion
            assert len(loaded.tokens) == len(sample_result.tokens)

    def test_to_openai_format(self, sample_result: CompletionResult):
        """Should export to OpenAI-compatible format."""
        openai_format = sample_result.to_openai_format()
        assert "content" in openai_format
        assert len(openai_format["content"]) == 2

        first_token = openai_format["content"][0]
        assert "token" in first_token
        assert "logprob" in first_token
        assert "top_logprobs" in first_token
        assert "bytes" in first_token

    def test_str_format(self, sample_result: CompletionResult):
        """Should produce human-readable string."""
        result_str = str(sample_result)
        assert "Prompt:" in result_str
        assert "Completion:" in result_str
        assert "Perplexity:" in result_str
        assert "Paris" in result_str


class TestTokenLogprobs:
    """Tests for TokenLogprobs model (scoring existing text)."""

    def test_basic_creation(self):
        """Should create with text and token scores."""
        tokens = [
            TokenLogprob(token="Hello", token_id=1, logprob=-0.5),
            TokenLogprob(token=" world", token_id=2, logprob=-0.3),
        ]
        result = TokenLogprobs(
            text="Hello world", tokens=tokens, total_logprob=-0.8, perplexity=1.5
        )
        assert result.text == "Hello world"
        assert len(result.tokens) == 2
        assert result.total_logprob == -0.8

    def test_json_export(self):
        """Should export to JSON."""
        result = TokenLogprobs(
            text="test",
            tokens=[TokenLogprob(token="test", token_id=1, logprob=-0.5)],
            total_logprob=-0.5,
            perplexity=1.65,
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["text"] == "test"


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_ranking_and_best(self):
        """Should rank continuations correctly."""
        continuations = [
            TokenLogprobs(
                text=" London",
                tokens=[TokenLogprob(token=" London", token_id=1, logprob=-2.0)],
                total_logprob=-2.0,
                perplexity=7.39,
            ),
            TokenLogprobs(
                text=" Paris",
                tokens=[TokenLogprob(token=" Paris", token_id=2, logprob=-0.5)],
                total_logprob=-0.5,
                perplexity=1.65,
            ),
            TokenLogprobs(
                text=" Berlin",
                tokens=[TokenLogprob(token=" Berlin", token_id=3, logprob=-1.5)],
                total_logprob=-1.5,
                perplexity=4.48,
            ),
        ]
        result = ComparisonResult(
            prompt="The capital of France is",
            continuations=continuations,
            ranking=[1, 2, 0],  # Paris, Berlin, London
        )

        assert result.best.text == " Paris"
        assert result.ranking[0] == 1

    def test_empty_ranking_raises(self):
        """Should raise error if no continuations."""
        result = ComparisonResult(prompt="test", continuations=[], ranking=[])
        with pytest.raises(ValueError, match="No continuations"):
            _ = result.best

    def test_json_export(self):
        """Should export to JSON."""
        result = ComparisonResult(
            prompt="test",
            continuations=[
                TokenLogprobs(
                    text="a",
                    tokens=[TokenLogprob(token="a", token_id=1, logprob=-0.5)],
                    total_logprob=-0.5,
                    perplexity=1.65,
                )
            ],
            ranking=[0],
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["prompt"] == "test"
        assert len(data["continuations"]) == 1
