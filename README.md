# Verifier Primacy

**Constrained decoding via logit-level rule verification for open-weight LLMs.**

> Verify at generation time. Don't pray and validate after.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

Traditional approach:
```
generate() → validate() → retry if invalid → 😭
```

This approach:
```
logits → mask_invalid() → sample() → guaranteed valid ✅
```

The model's logits are a probability distribution over tokens. We constrain that distribution to only valid tokens **before sampling**. Invalid tokens get `-inf` logits. They literally cannot be sampled.

**Why open weights?** API models hide their logits. You can't verify what you can't see.

## Installation

```bash
# Basic (numpy only)
pip install verifier-primacy

# With MLX (Apple Silicon)
pip install verifier-primacy[mlx]

# With vLLM (Production/GPU)
pip install verifier-primacy[vllm]

# Development
pip install verifier-primacy[dev]
```

Or with uv:
```bash
uv add verifier-primacy
uv add verifier-primacy --extra mlx
```

## Quick Start

```python
from verifier_primacy import create_json_verifier, FieldSpec

# Define your schema
schema = [
    FieldSpec(name="action", type="string", enum=["search", "create", "delete"]),
    FieldSpec(name="target", type="string"),
    FieldSpec(name="count", type="integer", min_value=0, max_value=100),
]

# Create verifier
verifier = create_json_verifier(tokenizer, schema)

# In generation loop:
for _ in range(max_tokens):
    logits = model(tokens)
    logits = verifier.apply(logits, state)  # ← THE KEY STEP
    next_token = sample(logits)
    state = verifier.update_state(state, next_token)
```

## Core Features

### 1. Per-Field Confidence Scores

```python
from verifier_primacy import entropy_confidence, top_k_gap

# Analyze confidence at each generation step
confidence = entropy_confidence(logits)  # [0, 1]

# Or use top-k gap for uncertainty between choices
gap_confidence = top_k_gap(logits, k=2)
```

### 2. Verification Primitives

```python
from verifier_primacy import FieldSpec, check_range, check_fuzzy_match

schema = [
    FieldSpec(
        name="amount",
        type="number",
        min_value=0,
        max_value=100000,
    ),
    FieldSpec(
        name="vendor",
        type="string",
        # Custom checks
        checks=[lambda v: check_fuzzy_match(v, known_vendors, threshold=0.8)]
    ),
]
```

### 3. Human Routing

```python
from verifier_primacy import Router, RoutingThresholds

router = Router(
    thresholds=RoutingThresholds(
        pass_threshold=0.9,    # Auto-accept above this
        reject_threshold=0.5,  # Auto-reject below this
    )
)

result = router.decide(extracted_data, confidence_scores, schema)

match result.decision:
    case RoutingDecision.PASS:
        save_to_db(extracted_data)
    case RoutingDecision.REVIEW:
        send_to_human(extracted_data, result.flagged_fields)
    case RoutingDecision.REJECT:
        log_failure(result.validation_errors)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LogitsVerifier                            │
├─────────────────────────────────────────────────────────────┤
│  VocabAnalyzer           │  Rules                           │
│  ┌───────────────────┐   │  ┌────────────────────────────┐  │
│  │ Token → Classes   │   │  │ JSONStructureRule          │  │
│  │ Precomputed masks │   │  │ SchemaRule                 │  │
│  └───────────────────┘   │  │ CustomRule (yours)         │  │
│                          │  └────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ParserState (FSM)                                          │
│  - in_string, in_object, depth, expected_type, etc.         │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Vocab analysis happens once at init. During generation, we just do mask lookups and boolean ops. Fast.

## Backends

### MLX (Apple Silicon)

```python
from verifier_primacy.backends import MLXBackend

backend = MLXBackend(model_path="mlx-community/Llama-3.2-3B-Instruct-4bit")
```

### vLLM (Production)

```python
from verifier_primacy.backends import VLLMBackend

backend = VLLMBackend(model="meta-llama/Llama-3.1-8B-Instruct")
```

## Custom Rules

```python
from verifier_primacy import VerificationRule, ParserState
import numpy as np

class NoSwearingRule(VerificationRule):
    def __init__(self, vocab, bad_words: list[str]):
        super().__init__(vocab)
        self.bad_token_ids = self._find_bad_tokens(bad_words)
    
    def get_allowed_mask(self, state: ParserState) -> np.ndarray:
        mask = np.ones(self.vocab.vocab_size, dtype=np.float32)
        mask[self.bad_token_ids] = 0.0
        return mask
```

## Performance

For a 32k vocab with 3 rules:

| Operation | Time | Notes |
|-----------|------|-------|
| Vocab analysis (init) | ~50ms | One-time cost |
| Per-token verification | ~20µs | Just mask ops |
| Mask combination | ~5µs | Vectorized |

Overhead is negligible compared to model forward pass.

## Development

```bash
# Clone
git clone https://github.com/andrewBatutin/verifier_primacy
cd verifier_primacy

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Type check
uv run pyright

# Format
uv run ruff format .
```

### Claude Code

This project is optimized for Claude Code. Key commands:

- `/project:test` - Run test suite
- `/project:bench` - Run benchmarks
- `/project:demo` - Run demo
- `/project:release 0.2.0` - Prepare release

## Related Work

- [Outlines](https://github.com/outlines-dev/outlines) - Similar idea, more mature
- [Instructor](https://github.com/jxnl/instructor) - Structured outputs via retries
- [Guidance](https://github.com/guidance-ai/guidance) - Microsoft's approach
- [LMQL](https://lmql.ai/) - Query language approach

This is a minimal, MLX-native implementation focused on:
1. **Transparency** - See exactly what's being verified
2. **Speed** - Minimal overhead per token
3. **Composability** - Mix and match rules

---

*Built with the belief that AI systems need verification layers, not better prayers.*

## License

MIT
