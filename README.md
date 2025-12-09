# Logits Verifier

**Constrained decoding via logit-level rule verification.**

> Verify at generation time. Don't pray and validate after.

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

This is the **Verifier Primacy** framework in action: build verification into the generation process itself.

## Installation

```bash
# Requires Apple Silicon for MLX
uv sync
```

## Quick Start

```python
from logits_verifier import create_json_verifier, FieldSpec

# Define your parameter schema
schema = [
    FieldSpec(name="action", type="string", enum=["search", "create", "delete"]),
    FieldSpec(name="target", type="string"),
    FieldSpec(name="count", type="integer"),
]

# Create verifier
verifier = create_json_verifier(tokenizer, schema)

# In generation loop:
logits = model(tokens)
logits = verifier.apply(logits, state)        # <- THE KEY STEP
next_token = sample(logits)
state = verifier.update_state(state, next_token)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LogitsVerifier                        │
├─────────────────────────────────────────────────────────┤
│  VocabAnalyzer          │  Rules                        │
│  ┌──────────────────┐   │  ┌─────────────────────────┐  │
│  │ Token → Classes  │   │  │ JSONStructureRule       │  │
│  │ Precomputed masks│   │  │ SchemaRule              │  │
│  └──────────────────┘   │  │ CustomRule (yours)      │  │
│                         │  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  ParserState (FSM)                                      │
│  - in_string, in_object, depth, expected_type, etc.    │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** Vocab analysis happens once at init. During generation, we just do mask lookups and boolean ops. Fast.

## Components

### VocabAnalyzer
Precomputes token classifications for the entire vocabulary:
- `DIGIT`, `ALPHA`, `WHITESPACE` - character types
- `LBRACE`, `RBRACE`, `QUOTE`, `COLON` - JSON structural
- `BOOL_TRUE`, `BOOL_FALSE`, `NULL` - JSON values

Creates boolean masks per class. Combining masks = OR operation.

### VerificationRule (Abstract)
Interface for custom rules:
```python
class MyRule(VerificationRule):
    def get_allowed_mask(self, state: ParserState, vocab: VocabAnalyzer) -> mx.array:
        # Return mask: 1.0 = allowed, 0.0 = forbidden
        ...
```

### Built-in Rules

**JSONStructureRule**: Enforces valid JSON syntax
- After `{` → expect `"` or `}`
- After `:` → expect value
- Inside string → expect content or closing `"`

**SchemaRule**: Enforces type constraints
- If expecting `integer` → only digits allowed
- If expecting `boolean` → only `true`/`false`
- Field name validation (TODO: prefix matching)

## Example: Tool Calling

```python
# Schema for tool parameters
schema = [
    FieldSpec(name="tool", type="string", enum=["web_search", "calculator", "file_read"]),
    FieldSpec(name="query", type="string"),
    FieldSpec(name="limit", type="integer"),
]

verifier = create_json_verifier(tokenizer, schema)

# Model CANNOT hallucinate invalid tool names
# Model CANNOT put string where integer expected
# Model CANNOT produce malformed JSON
```

## Performance Notes

1. **Vocab analysis is O(V)** where V = vocab size. Done once at init.
2. **Per-token verification is O(R)** where R = number of rules. Just mask ops.
3. **Mask combination is O(V)** but vectorized - fast on MLX.

For a 32k vocab with 3 rules, overhead is negligible compared to model forward pass.

## Why MLX?

- Native Apple Silicon support
- Lazy evaluation plays nice with our mask ops
- Same API as PyTorch, easy to port

But the verifier design is framework-agnostic. Swap `mx.array` for `torch.Tensor` and it works.

## TODO

- [ ] Prefix-constrained string generation (force specific field names)
- [ ] Recursive schema support (nested objects)
- [ ] Streaming integration with mlx-lm generate
- [ ] Benchmark against outlines/guidance

## Related Work

- [Outlines](https://github.com/outlines-dev/outlines) - Similar idea, more mature
- [Guidance](https://github.com/guidance-ai/guidance) - Microsoft's take
- [LMQL](https://lmql.ai/) - Query language approach

This is a minimal, MLX-native implementation for learning and experimentation.

---

*Built with the belief that AI systems need verification layers, not better prayers.*
