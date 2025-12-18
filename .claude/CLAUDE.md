# Verifier Primacy - Claude Code Context

## Project Philosophy

**Core thesis:** Verify at generation time, don't pray and validate after.

Traditional: `generate() → validate() → retry if invalid → 😭`
This approach: `logits → mask_invalid() → sample() → guaranteed valid ✅`

This is constrained decoding via logit-level verification for open-weight models.

## Tech Stack

- **Language:** Python 3.11+
- **Package Manager:** uv
- **Inference Backends:** MLX (Apple Silicon), vLLM (production)
- **Schema Validation:** Pydantic v2
- **Testing:** pytest
- **Type Checking:** pyright

## Project Structure

```
src/verifier_primacy/
├── core/           # Confidence scoring, primitives, routing logic
├── backends/       # MLX and vLLM inference wrappers
├── rules/          # Verification rules (JSON structure, schema, custom)
└── schemas/        # Pydantic integration
```

## Commands

```bash
# Development
uv sync                          # Install dependencies
uv run pytest                    # Run tests
uv run pytest tests/evals/       # Run evaluation suite
uv run pyright                   # Type check

# Demos
uv run python examples/basic_extraction.py
uv run python examples/tool_calling.py

# Benchmarks
uv run python benchmarks/bench_overhead.py
uv run python benchmarks/bench_accuracy.py
```

## Code Patterns

### Confidence Scoring
All confidence methods return values in [0, 1] range:
- `entropy_confidence()` - Based on logit distribution entropy
- `top_k_gap()` - Gap between top-1 and top-k probability
- `calibrated_confidence()` - Adjusted for model calibration

### Verification Rules
All rules inherit from `VerificationRule` ABC:
```python
class MyRule(VerificationRule):
    def get_allowed_mask(self, state: ParserState, vocab: VocabAnalyzer) -> Array:
        # Return mask: 1.0 = allowed, 0.0 = forbidden
        ...
```

### Backend Interface
All backends implement `InferenceBackend` protocol:
```python
class InferenceBackend(Protocol):
    def generate_with_logits(self, tokens: list[int]) -> tuple[int, Array]: ...
    def get_vocab_size(self) -> int: ...
```

## Testing Strategy

- Unit tests in `tests/` - run on every commit
- Evals in `tests/evals/` - run weekly, track regression
- Benchmarks in `benchmarks/` - run before releases

## Do Not

- Do not use API-based models (OpenAI, Anthropic) - we need logit access
- Do not add dependencies without updating pyproject.toml
- Do not commit without passing `uv run pytest` and `uv run pyright`
- Do not modify `src/verifier_primacy/backends/base.py` interface without updating both backends
- Do not use `print()` for logging - use proper Python `logging` module (`logger = logging.getLogger(__name__)`)

## Subagents

Use subagents for parallel work:
- `@reviewer` - Code review before commits
- `@benchmarker` - Performance testing
- `@docs-writer` - Documentation generation

## Additional Context

@docs/architecture.md
@docs/api.md
