Run the test suite for verifier_primacy:

1. Run all unit tests with verbose output
2. If any tests fail, analyze the failure and suggest fixes
3. Report coverage if available

```bash
uv run pytest -v --tb=short
```

If tests pass, summarize results. If tests fail, identify the root cause and propose a fix.
