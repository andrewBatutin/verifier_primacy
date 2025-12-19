#!/usr/bin/env python3
"""Example: Exploring logprobs with local MLX models.

This example demonstrates the VALUE that logprobs provide:
1. Confidence detection - know when to trust model output
2. Alternative insights - see what model almost said
3. Quality assessment - one-glance reliability scores
4. Export to JSON for downstream processing

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
            format_alternatives_insight,
            format_confidence_report,
            format_quality_summary,
            list_models,
        )
    except ImportError as e:
        logger.error("Error: %s", e)
        logger.error("Please install MLX dependencies:")
        logger.error("  pip install verifier-primacy[mlx]")
        return

    # Show available models
    logger.info("=" * 60)
    logger.info("MLX Logprobs Explorer - See What Your Model Is Thinking")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Available models (recommended first):")
    for i, model in enumerate(list_models()):
        rec = " [RECOMMENDED]" if i == 0 else ""
        logger.info("  - %s%s", model, rec)
    logger.info("")

    # Use recommended model for better quality
    model_id = "mlx-community/Qwen3-4B-4bit"
    logger.info("Loading: %s", model_id)
    logger.info("(First run downloads ~2GB, subsequent runs are fast)")
    logger.info("")

    try:
        explorer = LogprobsExplorer.from_pretrained(model_id)
    except Exception as e:
        logger.error("Could not load model: %s", e)
        logger.error("Falling back to smaller model...")
        try:
            model_id = "mlx-community/Qwen1.5-1.8B-Chat-4bit"
            explorer = LogprobsExplorer.from_pretrained(model_id)
        except Exception as e2:
            logger.error("Could not load fallback model: %s", e2)
            logger.error("Try: huggingface-cli download %s", model_id)
            return

    # ==========================================================================
    # VALUE DEMO 1: Quality Summary - One glance reliability check
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALUE 1: Quality Summary - Should I Trust This Output?")
    logger.info("=" * 60)

    prompt = "The capital of France is"
    result = explorer.complete(prompt, max_tokens=5, top_k=5)

    logger.info("")
    logger.info("%s", format_quality_summary(result))

    # ==========================================================================
    # VALUE DEMO 2: Confidence Report - Where is model uncertain?
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALUE 2: Confidence Report - Token-by-Token Trust Levels")
    logger.info("=" * 60)

    logger.info("")
    logger.info("%s", format_confidence_report(result))

    # ==========================================================================
    # VALUE DEMO 3: Alternative Paths - What else did model consider?
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALUE 3: Alternative Paths - The Roads Not Taken")
    logger.info("=" * 60)

    logger.info("")
    logger.info("%s", format_alternatives_insight(result))

    # ==========================================================================
    # VALUE DEMO 4: Hallucination Detection - Uncertain generation
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALUE 4: Hallucination Risk - Spotting Uncertain Claims")
    logger.info("=" * 60)

    # Generate something the model might be uncertain about
    uncertain_prompt = "The 47th President of the United States is"
    uncertain_result = explorer.complete(uncertain_prompt, max_tokens=10, top_k=5)

    logger.info("")
    logger.info("Prompt: %r", uncertain_prompt)
    logger.info("")
    logger.info("%s", format_quality_summary(uncertain_result))
    logger.info("")
    logger.info("%s", format_confidence_report(uncertain_result))

    # ==========================================================================
    # VALUE DEMO 5: Compare Options - Which continuation is most natural?
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALUE 5: Compare Options - Rank by Model Confidence")
    logger.info("=" * 60)

    comparison = explorer.compare_continuations(
        prompt="The best way to learn programming is",
        continuations=[
            " through practice",
            " by reading books",
            " with online courses",
            " impossible",
        ],
    )

    logger.info("")
    logger.info("Prompt: %r", comparison.prompt)
    logger.info("")
    logger.info("Model's preference ranking:")
    for rank, idx in enumerate(comparison.ranking):
        cont = comparison.continuations[idx]
        marker = " <- MODEL'S CHOICE" if rank == 0 else ""
        logger.info(
            "  %d. %r (perplexity: %.2f)%s",
            rank + 1,
            cont.text,
            cont.perplexity,
            marker,
        )

    # ==========================================================================
    # JSON Export for programmatic use
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPORT: JSON for Downstream Processing")
    logger.info("=" * 60)

    result.save_json("logprobs_output.json")
    logger.info("")
    logger.info("Saved to: logprobs_output.json")
    logger.info("")
    logger.info("Use in your pipeline:")
    logger.info("  data = result.to_dict()")
    logger.info("  for token in data['tokens']:")
    logger.info("      if token['chosen']['prob'] < 0.4:")
    logger.info("          flag_for_review(token)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("KEY TAKEAWAY: Logprobs let you KNOW when to trust LLM output")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
