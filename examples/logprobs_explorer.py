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

import logging

# Configure logging for this example
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


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
        logger.error("Error: %s", e)
        logger.error("Please install MLX dependencies:")
        logger.error("  pip install verifier-primacy[mlx]")
        return

    # Show available models
    logger.info("=" * 60)
    logger.info("Available MLX Models")
    logger.info("=" * 60)
    for model in list_models():
        logger.info("  - %s", model)
    logger.info("")

    # Choose a model (using a small one for the example)
    model_id = "mlx-community/Llama-3.2-1B-4bit"
    logger.info("Loading model: %s", model_id)
    logger.info("(This may take a moment on first run as the model downloads...)")
    logger.info("")

    try:
        explorer = LogprobsExplorer.from_pretrained(model_id)
    except Exception as e:
        logger.error("Could not load model: %s", e)
        logger.error("If the model isn't downloaded, try running:")
        logger.error("  huggingface-cli download %s", model_id)
        return

    # Example 1: Basic completion with logprobs
    logger.info("=" * 60)
    logger.info("Example 1: Basic Completion with Logprobs")
    logger.info("=" * 60)

    prompt = "The capital of France is"
    result = explorer.complete(prompt, max_tokens=5, top_k=5)

    # Human-readable output
    print_logprobs(result)

    # Example 2: Compact format
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Compact Format")
    logger.info("=" * 60)
    logger.info("%s", format_compact(result))

    # Example 3: JSON export
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: JSON Export")
    logger.info("=" * 60)

    # Get as dict for processing
    data = result.to_dict()
    logger.info("Number of tokens: %d", len(data["tokens"]))
    logger.info("Total logprob: %.4f", data["total_logprob"])
    logger.info("Perplexity: %.2f", data["perplexity"])

    # Save to file
    result.save_json("logprobs_output.json")
    logger.info("\nSaved to: logprobs_output.json")

    # Show raw JSON snippet
    logger.info("\nJSON preview:")
    json_str = result.to_json(indent=2)
    # Show first 500 chars
    if len(json_str) > 500:
        logger.info("%s\n...", json_str[:500])
    else:
        logger.info("%s", json_str)

    # Example 4: Compare continuations
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Compare Continuations")
    logger.info("=" * 60)

    comparison = explorer.compare_continuations(
        prompt="The best programming language is",
        continuations=[" Python", " JavaScript", " Rust", " Go"],
    )

    logger.info("Prompt: %r\n", comparison.prompt)
    logger.info("Rankings (most likely first):")
    for rank, idx in enumerate(comparison.ranking):
        cont = comparison.continuations[idx]
        marker = " <-- most likely" if rank == 0 else ""
        logger.info(
            "  %d. %r: logprob=%.3f, perplexity=%.2f%s",
            rank + 1,
            cont.text,
            cont.total_logprob,
            cont.perplexity,
            marker,
        )

    logger.info("\nBest continuation: %r", comparison.best.text)

    # Example 5: Score existing text
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Score Existing Text")
    logger.info("=" * 60)

    score_result = explorer.get_logprobs(
        prompt="Machine learning is",
        continuation=" a subset of artificial intelligence",
    )
    logger.info("Text: %r", score_result.text)
    logger.info("Total logprob: %.4f", score_result.total_logprob)
    logger.info("Perplexity: %.2f", score_result.perplexity)
    logger.info("\nPer-token breakdown:")
    for token in score_result.tokens:
        logger.info("  '%s': %.2f%% (logprob: %.3f)", token.token, token.prob * 100, token.logprob)

    # Example 6: Find uncertain tokens
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Analyze Uncertainty")
    logger.info("=" * 60)

    # Generate a longer completion
    longer_result = explorer.complete(
        "In the year 2050, technology will",
        max_tokens=20,
        top_k=5,
    )

    uncertain = explorer.analyze_uncertainty(longer_result, threshold=0.5)
    logger.info("Generated: %r", longer_result.completion)
    logger.info("\nTokens with probability < 50%%: %d", len(uncertain))
    for token in uncertain[:5]:  # Show first 5
        logger.info("  Position %d: %s", token.position, token.chosen)
        if token.alternatives:
            logger.info("    Top alternative: %s", token.alternatives[0])

    # Example 7: OpenAI-compatible format
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: OpenAI-Compatible Format")
    logger.info("=" * 60)

    openai_format = result.to_openai_format()
    logger.info("OpenAI format structure:")
    logger.info("  - content: list of %d token objects", len(openai_format["content"]))
    if openai_format["content"]:
        first = openai_format["content"][0]
        logger.info("  - First token: %r", first["token"])
        logger.info("  - Logprob: %.4f", first["logprob"])
        logger.info("  - Top alternatives: %d", len(first["top_logprobs"]))

    logger.info("\n" + "=" * 60)
    logger.info("Done! Check logprobs_output.json for the full output.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
