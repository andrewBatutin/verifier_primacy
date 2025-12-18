"""Pytest configuration and fixtures for verifier_primacy tests."""

import numpy as np
import pytest
from pydantic import BaseModel

from verifier_primacy.core.primitives import FieldSpec


class MockTokenizer:
    """Mock tokenizer for testing without real model."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab_size = vocab_size
        # Create a simple vocabulary
        self._vocab = {i: chr(32 + (i % 95)) for i in range(vocab_size)}

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str) -> list[int]:
        return [ord(c) - 32 for c in text if 0 <= ord(c) - 32 < self._vocab_size]


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Provide a mock tokenizer for testing."""
    return MockTokenizer(vocab_size=1000)


@pytest.fixture
def sample_schema() -> list[FieldSpec]:
    """Provide a sample schema for testing."""
    return [
        FieldSpec(
            name="action",
            type="string",
            enum=["search", "create", "delete"],
            description="The action to perform",
        ),
        FieldSpec(
            name="target",
            type="string",
            description="Target of the action",
        ),
        FieldSpec(
            name="count",
            type="integer",
            min_value=0,
            max_value=100,
            description="Number of items",
        ),
    ]


@pytest.fixture
def invoice_schema() -> list[FieldSpec]:
    """Provide an invoice extraction schema for testing."""
    return [
        FieldSpec(
            name="vendor",
            type="string",
            description="Vendor name",
        ),
        FieldSpec(
            name="amount",
            type="number",
            min_value=0,
            description="Invoice amount",
        ),
        FieldSpec(
            name="currency",
            type="string",
            enum=["USD", "EUR", "GBP"],
            description="Currency code",
        ),
        FieldSpec(
            name="date",
            type="string",
            pattern=r"\d{4}-\d{2}-\d{2}",
            description="Invoice date in YYYY-MM-DD format",
        ),
    ]


class Invoice(BaseModel):
    """Pydantic model for invoice extraction."""

    vendor: str
    amount: float
    currency: str
    date: str


class ToolCall(BaseModel):
    """Pydantic model for tool calling."""

    action: str
    target: str
    count: int = 1


@pytest.fixture
def invoice_model() -> type[Invoice]:
    """Provide Invoice Pydantic model."""
    return Invoice


@pytest.fixture
def tool_call_model() -> type[ToolCall]:
    """Provide ToolCall Pydantic model."""
    return ToolCall


@pytest.fixture
def uniform_logits() -> np.ndarray:
    """Provide uniform logits (low confidence)."""
    return np.zeros(1000, dtype=np.float32)


@pytest.fixture
def peaked_logits() -> np.ndarray:
    """Provide peaked logits (high confidence)."""
    logits = np.full(1000, -100.0, dtype=np.float32)
    logits[42] = 0.0  # One high value
    return logits


@pytest.fixture
def mixed_logits() -> np.ndarray:
    """Provide logits with top-2 competition (medium confidence)."""
    logits = np.full(1000, -100.0, dtype=np.float32)
    logits[42] = 0.0
    logits[43] = -0.1  # Close second
    return logits


# Skip markers for backend-specific tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "mlx: mark test as requiring MLX")
    config.addinivalue_line("markers", "vllm: mark test as requiring vLLM")
    config.addinivalue_line("markers", "slow: mark test as slow-running")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available backends."""
    # Check MLX availability
    try:
        import mlx.core  # noqa: F401

        has_mlx = True
    except ImportError:
        has_mlx = False

    # Check vLLM availability
    try:
        import torch
        import vllm  # noqa: F401

        has_vllm = torch.cuda.is_available()
    except ImportError:
        has_vllm = False

    skip_mlx = pytest.mark.skip(reason="MLX not available")
    skip_vllm = pytest.mark.skip(reason="vLLM not available or no CUDA")

    for item in items:
        if "mlx" in item.keywords and not has_mlx:
            item.add_marker(skip_mlx)
        if "vllm" in item.keywords and not has_vllm:
            item.add_marker(skip_vllm)
