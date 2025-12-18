#!/usr/bin/env python3
"""Example: Exploring logprobs with local MLX models.

This example demonstrates how to use the LogprobsExplorer to:
1. Generate text with log-probability information
2. Inspect token-by-token probabilities
3. Compare multiple continuations
4. Export results to JSON

Requirements:
    pip install verifier-primacy[mlx]

Usage:
    python examples/logprobs_explorer.py
"""

from __future__ import annotations


def main():
    """Run the logprobs exploration examples."""
    # Import here so we can show helpful error if MLX not installed
    try:
        from verifier_primacy.logprobs import (
            LogprobsExplorer,
            format_compact,
            list_models,
            print_logprobs,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install MLX dependencies:")
        print("  pip install verifier-primacy[mlx]")
        return

    # Show available models
    print("=" * 60)
    print("Available MLX Models")
    print("=" * 60)
    for model in list_models():
        print(f"  - {model}")
    print()

    # Choose a model (using a small one for the example)
    model_id = "mlx-community/Llama-3.2-1B-4bit"
    print(f"Loading model: {model_id}")
    print("(This may take a moment on first run as the model downloads...)")
    print()

    try:
        explorer = LogprobsExplorer.from_pretrained(model_id)
    except Exception as e:
        print(f"Could not load model: {e}")
        print("\nIf the model isn't downloaded, try running:")
        print(f"  huggingface-cli download {model_id}")
        return

    # Example 1: Basic completion with logprobs
    print("=" * 60)
    print("Example 1: Basic Completion with Logprobs")
    print("=" * 60)

    prompt = "The capital of France is"
    result = explorer.complete(prompt, max_tokens=5, top_k=5)

    # Human-readable output
    print_logprobs(result)

    # Example 2: Compact format
    print("\n" + "=" * 60)
    print("Example 2: Compact Format")
    print("=" * 60)
    print(format_compact(result))

    # Example 3: JSON export
    print("\n" + "=" * 60)
    print("Example 3: JSON Export")
    print("=" * 60)

    # Get as dict for processing
    data = result.to_dict()
    print(f"Number of tokens: {len(data['tokens'])}")
    print(f"Total logprob: {data['total_logprob']:.4f}")
    print(f"Perplexity: {data['perplexity']:.2f}")

    # Save to file
    result.save_json("logprobs_output.json")
    print("\nSaved to: logprobs_output.json")

    # Show raw JSON snippet
    print("\nJSON preview:")
    json_str = result.to_json(indent=2)
    # Show first 500 chars
    if len(json_str) > 500:
        print(json_str[:500] + "\n...")
    else:
        print(json_str)

    # Example 4: Compare continuations
    print("\n" + "=" * 60)
    print("Example 4: Compare Continuations")
    print("=" * 60)

    comparison = explorer.compare_continuations(
        prompt="The best programming language is",
        continuations=[" Python", " JavaScript", " Rust", " Go"],
    )

    print(f"Prompt: {comparison.prompt!r}\n")
    print("Rankings (most likely first):")
    for rank, idx in enumerate(comparison.ranking):
        cont = comparison.continuations[idx]
        marker = " <-- most likely" if rank == 0 else ""
        print(
            f"  {rank + 1}. {cont.text!r}: "
            f"logprob={cont.total_logprob:.3f}, "
            f"perplexity={cont.perplexity:.2f}{marker}"
        )

    print(f"\nBest continuation: {comparison.best.text!r}")

    # Example 5: Score existing text
    print("\n" + "=" * 60)
    print("Example 5: Score Existing Text")
    print("=" * 60)

    score_result = explorer.get_logprobs(
        prompt="Machine learning is",
        continuation=" a subset of artificial intelligence",
    )
    print(f"Text: {score_result.text!r}")
    print(f"Total logprob: {score_result.total_logprob:.4f}")
    print(f"Perplexity: {score_result.perplexity:.2f}")
    print("\nPer-token breakdown:")
    for token in score_result.tokens:
        print(f"  '{token.token}': {token.prob:.2%} (logprob: {token.logprob:.3f})")

    # Example 6: Find uncertain tokens
    print("\n" + "=" * 60)
    print("Example 6: Analyze Uncertainty")
    print("=" * 60)

    # Generate a longer completion
    longer_result = explorer.complete(
        "In the year 2050, technology will",
        max_tokens=20,
        top_k=5,
    )

    uncertain = explorer.analyze_uncertainty(longer_result, threshold=0.5)
    print(f"Generated: {longer_result.completion!r}")
    print(f"\nTokens with probability < 50%: {len(uncertain)}")
    for token in uncertain[:5]:  # Show first 5
        print(f"  Position {token.position}: {token.chosen}")
        if token.alternatives:
            print(f"    Top alternative: {token.alternatives[0]}")

    # Example 7: OpenAI-compatible format
    print("\n" + "=" * 60)
    print("Example 7: OpenAI-Compatible Format")
    print("=" * 60)

    openai_format = result.to_openai_format()
    print("OpenAI format structure:")
    print(f"  - content: list of {len(openai_format['content'])} token objects")
    if openai_format["content"]:
        first = openai_format["content"][0]
        print(f"  - First token: {first['token']!r}")
        print(f"  - Logprob: {first['logprob']:.4f}")
        print(f"  - Top alternatives: {len(first['top_logprobs'])}")

    print("\n" + "=" * 60)
    print("Done! Check logprobs_output.json for the full output.")
    print("=" * 60)


if __name__ == "__main__":
    main()
