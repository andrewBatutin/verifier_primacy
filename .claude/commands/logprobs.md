Analyze LLM token probabilities, confidence, and alternatives using local MLX models.

Arguments: $ARGUMENTS (prompt and optional flags)

## Modes

- **Complete** (default): Generate text and show confidence/alternatives analysis
- **Score**: Evaluate how likely existing text is given a prompt
- **Compare**: Rank multiple continuations by model likelihood

## Usage

```bash
# Complete mode - generate and analyze
/logprobs "The capital of France is"

# Score mode - evaluate existing text
/logprobs --mode score "The capital of France is" --continuation " Paris"

# Compare mode - rank alternatives
/logprobs --mode compare "The best way to learn programming is" --alternatives " through practice" " by reading books" " impossible"

# With options
/logprobs --max-tokens 30 --top-k 10 "Once upon a time"
/logprobs --model mlx-community/Llama-3.2-1B-4bit "Test prompt"
/logprobs --json "Output as JSON"
```

## Run the Analysis

```bash
uv run python scripts/logprobs_cli.py $ARGUMENTS
```

## Interpreting Results

After running, explain the results to the user:

1. **Quality Summary**: Overall reliability grade based on perplexity
   - A (Excellent): perplexity < 1.5 - very natural, trustworthy
   - B (Good): perplexity 1.5-3.0 - natural output
   - C (Moderate): perplexity 3.0-6.0 - acceptable but verify
   - D/F (Poor): perplexity > 6.0 - model struggled, review carefully

2. **Confidence Report**: Per-token trust levels
   - High (>70%): Model is certain about this token
   - Medium (40-70%): Acceptable but worth verifying
   - Low (<40%): Potential hallucination risk - flag for review

3. **Alternatives**: What the model almost said at each position
   - Semantically similar alternatives = model was confident in meaning
   - Semantically different alternatives = model was uncertain about direction

## Deep Analysis

For deeper interpretation of results, use the `@logprobs-analyst` subagent which can:
- Explain uncertainty patterns in context
- Identify hallucination risk areas
- Suggest prompt improvements to increase model confidence
