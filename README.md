# Verifier Primacy

Logprobs analysis toolkit for local LLMs on Apple Silicon.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Claude Code Skill

```
/logprobs "The capital of France is"
```

Analyze token probabilities, confidence scores, and alternatives from local MLX models.

### Modes

**Complete** (default) - Generate and analyze confidence
```
/logprobs "The capital of France is"
```

**Score** - Evaluate likelihood of existing text
```
/logprobs --mode score "The capital" --continuation " of France"
```

**Compare** - Rank multiple continuations
```
/logprobs --mode compare "The answer is" --alternatives " Paris" " London" " Berlin"
```

### Output Metrics

| Metric | What it tells you |
|--------|-------------------|
| **Perplexity** | Overall quality (A: <1.5, B: 1.5-3, C: 3-6, D: >6) |
| **Confidence %** | Per-token certainty (High >70%, Med 40-70%, Low <40%) |
| **Alternatives** | What the model almost said at each position |

### Example Output

```
Prompt: "The capital of France is"
Output: "**Paris**."

Perplexity: 1.01 (EXCELLENT)
Avg Confidence: 99% (HIGH)
Uncertain Tokens: 0 of 10 (0%)
```

### Options

- `--think` - Enable Qwen3 thinking mode (disabled by default)
- `--max-tokens N` - Limit generation length
- `--top-k N` - Number of alternatives to show
- `--json` - Output as JSON
- `--model <path>` - Use different MLX model

## Installation

```bash
# With uv (recommended)
uv sync

# Or pip
pip install -e ".[mlx]"
```

Requires Apple Silicon for MLX backend.

## Python API

```python
from verifier_primacy.logprobs import LogprobsExplorer

explorer = LogprobsExplorer.from_pretrained("mlx-community/Qwen3-4B-4bit")

# Generate with confidence analysis
result = explorer.complete("The capital of France is", max_tokens=20)
print(f"Output: {result.completion}")
print(f"Perplexity: {result.perplexity:.2f}")

# Score existing text
score = explorer.get_logprobs("Hello", " world")
print(f"Likelihood: {score.perplexity:.2f}")

# Compare alternatives
comparison = explorer.compare_continuations(
    "The best programming language is",
    [" Python", " JavaScript", " Rust"]
)
print(f"Model prefers: {comparison.best.text}")
```

## Coming Soon

- Constrained decoding via logit-level verification
- Schema-guided generation
- Human-in-the-loop routing

## Development

```bash
git clone https://github.com/andrewBatutin/verifier_primacy
cd verifier_primacy
uv sync
uv run pytest
```

## License

MIT
