---
paths:
  - "src/verifier_primacy/backends/mlx.py"
  - "**/mlx/**"
---

# MLX Backend Rules

## MLX-Specific Patterns

- Use `mx.array` not `np.array` for GPU operations
- Lazy evaluation: operations don't execute until needed
- Use `mx.eval()` to force computation when timing
- Memory is shared between CPU and GPU on Apple Silicon

## Common Pitfalls

```python
# BAD: Creates unnecessary copy
mask = mx.array(numpy_mask)

# GOOD: Direct construction
mask = mx.zeros(vocab_size)
mask[allowed_indices] = 1.0
```

## Performance Notes

- Vocab analysis masks should be computed once at init
- Use `mx.compile()` for hot paths
- Batch operations where possible
- Profile with `mx.metal.start_capture()` / `stop_capture()`

## Testing MLX Code

Always check for MLX availability:
```python
import pytest

mlx = pytest.importorskip("mlx.core")
```
