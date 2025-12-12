# Style Verifier

Prevent LLMs from generating banned phrases using logit masking at generation time.

## Overview

Style Verifier intercepts the text generation process and masks logits for tokens that would continue a banned phrase sequence. Unlike post-generation filtering, banned phrases are **impossible** to generate.

## Quick Start

```bash
# Run the A/B test app
cd style_verifier
streamlit run app.py
```

## Usage

### Programmatic

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from style_verifier import StyleVerifier
from style_verifier.generate import make_logits_processor

# Load model
model, tokenizer = load("mlx-community/Qwen3-4B-4bit")

# Load verifier
verifier = StyleVerifier()

# Create logits processor
processor = make_logits_processor(verifier)

# Generate with masking
result = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Why most startups fail:",
    max_tokens=200,
    sampler=make_sampler(temp=0),
    logits_processors=[processor],
)
```

### Streamlit App

```bash
streamlit run app.py
```

Features:
- Side-by-side A/B comparison (raw vs filtered)
- Temperature and max tokens controls
- Banned phrase highlighting
- Real-time verification

## How It Works

1. **Tokenize banned phrases** - Each phrase is converted to a token sequence
2. **Monitor generation** - Track tokens as they're generated
3. **Prefix matching** - Check if recent tokens match any banned sequence prefix
4. **Mask logits** - Set logit of next banned token to `-inf`
5. **Impossible to sample** - Model cannot select masked tokens

```
Banned: "In this article" -> [641, 419, 4549]

Generated so far: [..., 641, 419]  (matches prefix "In this")
                          ↓
Block token 4549 ("article") -> logits[4549] = -inf
                          ↓
Model forced to choose different continuation
```

## File Structure

```
style_verifier/
├── __init__.py           # Package exports
├── banned.txt            # Banned phrases (one per line)
├── banned_tokens.json    # Tokenized sequences (auto-generated)
├── tokenize_bans.py      # Tokenization script
├── style_verifier.py     # Core StyleVerifier class
├── generate.py           # Generation utilities
├── app.py                # Streamlit A/B test app
└── test.sh               # Validation script
```

## Adding Banned Phrases

1. Edit `banned.txt`:
```
existing phrase
new banned phrase
another one
```

2. Regenerate tokens:
```bash
python tokenize_bans.py
```

This automatically creates both regular and space-prefixed variants to catch phrases appearing mid-sentence.

## Banned Phrases

Current list:
- `And here's what`
- `But here's the thing`
- `Here's what nobody`
- `I was wrong`
- `I used to think`
- `The real unlock is`
- `The real secret is`
- `Let me explain`
- `In this article`
- `In this thread`
- `is dead` / `are dead`
- `A study of 1000+ startups`

## Testing

```bash
# Run validation
./test.sh

# Manual test
python generate.py
```

## Requirements

- `mlx>=0.21.0`
- `mlx-lm>=0.19.0`
- `streamlit>=1.40.0` (for app)

All dependencies are in the parent `pyproject.toml`.
