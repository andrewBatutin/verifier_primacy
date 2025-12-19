---
name: logprobs
description: Analyze LLM token probabilities for confidence calibration. Complete, score, and compare modes for evaluating model uncertainty.
---

# Logprobs Skill

Analyze token-level confidence from local LLMs running on MLX backend.

## Modes

### Complete Mode (default)
Generate text and analyze confidence/alternatives.
```
/logprobs "The capital of France is"
```

### Score Mode
Evaluate likelihood of existing text.
```
/logprobs --mode score "The capital" --continuation " of France"
```

### Compare Mode
Rank multiple continuations.
```
/logprobs --mode compare "The answer is" --alternatives " Paris" " London" " Berlin"
```

## Output Metrics

- **Perplexity**: Lower = more confident (A: <1.5, B: 1.5-3, C: 3-6, D: >6)
- **Confidence %**: Token-level certainty (High >70%, Med 40-70%, Low <40%)
- **Uncertain tokens**: Flags where model hedges

## Options

- `--think`: Enable Qwen3 thinking mode (disabled by default)
- `--max-tokens N`: Limit generation length
- `--top-k N`: Number of alternatives to show
- `--json`: Output as JSON
