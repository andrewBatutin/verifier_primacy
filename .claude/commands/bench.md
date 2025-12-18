Run performance benchmarks for verifier_primacy:

1. Run overhead benchmark (how much latency does verification add?)
2. Run accuracy benchmark (does verification improve output quality?)
3. Compare against baseline (no verification)

```bash
uv run python benchmarks/bench_overhead.py
uv run python benchmarks/bench_accuracy.py
```

Summarize results in a table format:
- Tokens/second with and without verification
- Accuracy improvement percentage
- Memory overhead

Flag any regressions compared to previous runs if benchmark history exists.
