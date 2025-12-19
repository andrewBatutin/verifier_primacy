#!/usr/bin/env python3
"""CLI for logprobs analysis - used by Claude /logprobs skill.

This script provides a command-line interface for analyzing LLM token
probabilities, confidence scores, and alternative tokens.

Usage:
    # Complete mode (default): Generate and analyze
    python scripts/logprobs_cli.py "The capital of France is"

    # Score mode: Evaluate existing text
    python scripts/logprobs_cli.py --mode score "The capital" --continuation " of France"

    # Compare mode: Rank alternatives
    python scripts/logprobs_cli.py --mode compare "Best way:" --alternatives " A" " B" " C"
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run logprobs analysis based on CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze LLM logprobs for confidence and alternatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "The capital of France is"
  %(prog)s --mode score "Hello" --continuation " world"
  %(prog)s --mode compare "Best:" --alternatives " A" " B" " C"
  %(prog)s --model mlx-community/Llama-3.2-1B-4bit "Test prompt"
        """,
    )
    parser.add_argument("prompt", help="The prompt to analyze")
    parser.add_argument(
        "--mode",
        choices=["complete", "score", "compare"],
        default="complete",
        help="Analysis mode (default: complete)",
    )
    parser.add_argument(
        "--continuation",
        help="Text to score (required for score mode)",
    )
    parser.add_argument(
        "--alternatives",
        nargs="+",
        help="Alternatives to compare (required for compare mode)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-4B-4bit",
        help="Model to use (default: mlx-community/Qwen3-4B-4bit)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate in complete mode (default: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top alternatives to show (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (disabled by default)",
    )
    parser.add_argument(
        "--detect-boundaries",
        action="store_true",
        help="Detect and explore decision boundaries (tool vs text divergence)",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.1,
        help="Minimum probability for boundary detection (default: 0.1 = 10%%)",
    )

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "score" and not args.continuation:
        logger.error("Error: --continuation is required for score mode")
        return 1

    if args.mode == "compare" and not args.alternatives:
        logger.error("Error: --alternatives is required for compare mode")
        return 1

    # Import here to allow --help without loading model
    try:
        from verifier_primacy.logprobs import (
            LogprobsExplorer,
            format_alternatives_insight,
            format_comparison,
            format_confidence_report,
            format_quality_summary,
        )
    except ImportError as e:
        logger.error("Error: %s", e)
        logger.error("Please install MLX dependencies: pip install verifier-primacy[mlx]")
        return 1

    # Load model
    logger.info("Loading model: %s", args.model)
    logger.info("(First run may download the model)")
    logger.info("")

    try:
        explorer = LogprobsExplorer.from_pretrained(args.model)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return 1

    # Run analysis based on mode
    if args.mode == "complete":
        result = explorer.complete(
            args.prompt,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            disable_thinking=not args.think,
        )

        if args.detect_boundaries:
            # Import boundary detection
            from verifier_primacy.logprobs.boundaries import (
                filter_critical_boundaries,
                find_decision_boundaries,
            )
            from verifier_primacy.logprobs.formatters import (
                format_boundary_report,
                format_position_zero_analysis,
            )

            # Find boundaries
            boundaries = find_decision_boundaries(result, threshold=args.boundary_threshold)
            critical = filter_critical_boundaries(boundaries)

            # Generate alternative paths for critical boundaries
            alternative_results: dict[int, object] = {}
            for boundary in critical:
                # Get tokens up to this position, then force the alternative
                prefix_tokens = [t.chosen.token_id for t in result.tokens[: boundary.position]]
                prefix_tokens.append(boundary.alternative.token_id)

                alt_result = explorer.complete_from_tokens(
                    prompt=args.prompt,
                    prefix_token_ids=prefix_tokens,
                    max_tokens=args.max_tokens - len(prefix_tokens),
                    top_k=args.top_k,
                    disable_thinking=not args.think,
                )
                alternative_results[boundary.position] = alt_result

            # Output
            if args.json:
                import json

                print(
                    json.dumps(
                        {
                            "primary": result.to_dict(),
                            "boundaries": [
                                {
                                    "position": b.position,
                                    "chosen": b.chosen.token,
                                    "alternative": b.alternative.token,
                                    "type": b.boundary_type,
                                    "risk": b.risk_level,
                                }
                                for b in critical
                            ],
                            "alternative_paths": {
                                str(pos): r.to_dict() for pos, r in alternative_results.items()
                            },
                        },
                        indent=2,
                    )
                )
            else:
                print(format_quality_summary(result))
                print()
                # Always show position 0 analysis (first token decision)
                if result.tokens:
                    print(format_position_zero_analysis(result.tokens[0]))
                    print()
                print(format_boundary_report(critical, result, alternative_results))

        elif args.json:
            import json

            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_quality_summary(result))
            print()
            print(format_confidence_report(result))
            print()
            print(format_alternatives_insight(result))

    elif args.mode == "score":
        result = explorer.get_logprobs(args.prompt, args.continuation)

        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "text": result.text,
                        "perplexity": result.perplexity,
                        "total_logprob": result.total_logprob,
                        "tokens": [
                            {
                                "token": t.token,
                                "token_id": t.token_id,
                                "logprob": t.logprob,
                                "prob": t.prob,
                            }
                            for t in result.tokens
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print("=" * 50)
            print("SCORE ANALYSIS")
            print("=" * 50)
            print(f"Prompt: {args.prompt!r}")
            print(f"Continuation: {result.text!r}")
            print()
            print(f"Perplexity: {result.perplexity:.2f}")
            print(f"Total logprob: {result.total_logprob:.4f}")
            print()
            print("Per-token breakdown:")
            for i, token in enumerate(result.tokens):
                prob_bar = "#" * int(token.prob * 20)
                print(f"  {i + 1}. {token.token!r:15} prob={token.prob:.1%} {prob_bar}")

    elif args.mode == "compare":
        result = explorer.compare_continuations(args.prompt, args.alternatives)

        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "prompt": result.prompt,
                        "ranking": result.ranking,
                        "continuations": [
                            {
                                "text": c.text,
                                "perplexity": c.perplexity,
                                "total_logprob": c.total_logprob,
                            }
                            for c in result.continuations
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(format_comparison(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
