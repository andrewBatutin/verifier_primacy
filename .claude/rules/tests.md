---
paths:
  - "tests/**/*.py"
  - "test_*.py"
  - "*_test.py"
---

# Testing Rules

## Test Structure

```python
# test_<module>.py

import pytest
from verifier_primacy.core import confidence

class TestEntropyConfidence:
    """Tests for entropy_confidence function."""
    
    def test_uniform_distribution_low_confidence(self):
        """Uniform distribution should have low confidence."""
        # Arrange
        logits = mx.zeros(1000)  # Uniform
        
        # Act
        conf = confidence.entropy_confidence(logits)
        
        # Assert
        assert conf < 0.1
    
    def test_peaked_distribution_high_confidence(self):
        """Peaked distribution should have high confidence."""
        logits = mx.full(1000, -100.0)
        logits[42] = 0.0  # One high value
        
        conf = confidence.entropy_confidence(logits)
        
        assert conf > 0.9
```

## Fixtures

Define in `conftest.py`:
```python
@pytest.fixture
def sample_schema():
    class Invoice(BaseModel):
        amount: float
        vendor: str
    return Invoice

@pytest.fixture
def mock_tokenizer():
    # Return a minimal tokenizer for testing
    ...
```

## Parameterized Tests

Use for edge cases:
```python
@pytest.mark.parametrize("input,expected", [
    ([], 0.0),
    ([0.5], 1.0),
    ([0.5, 0.5], 0.5),
])
def test_confidence_edge_cases(input, expected):
    ...
```

## Evals vs Unit Tests

- `tests/` - Fast, deterministic, run on every commit
- `tests/evals/` - Slower, may use real models, run weekly

## Markers

```python
@pytest.mark.slow  # Skip with -m "not slow"
@pytest.mark.mlx   # Requires MLX
@pytest.mark.vllm  # Requires vLLM + GPU
```
