---
paths:
  - "src/verifier_primacy/backends/vllm.py"
  - "**/vllm/**"
---

# vLLM Backend Rules

## vLLM-Specific Patterns

- Use `SamplingParams` for generation config
- Access logits via `LogitsProcessor` callback
- Tensor operations use PyTorch

## Logits Processor Pattern

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.logits_processor import LogitsProcessor

class VerifierLogitsProcessor(LogitsProcessor):
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        # Apply verification mask
        mask = self.verifier.get_mask(self.state)
        scores = scores + mask  # -inf for forbidden tokens
        return scores
```

## Production Considerations

- vLLM runs as a server, not in-process
- Consider batching for throughput
- Memory management is handled by PagedAttention
- Profile with `py-spy` for CPU bottlenecks

## Testing vLLM Code

vLLM requires GPU - use fixtures that skip on CPU:
```python
import pytest

@pytest.fixture
def vllm_model():
    pytest.importorskip("vllm")
    if not torch.cuda.is_available():
        pytest.skip("vLLM requires CUDA")
    # ... setup ...
```
