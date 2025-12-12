"""Build token mappings for categories and redirects."""

import json
from pathlib import Path

from mlx_lm import load

from categories import CATEGORIES
from patterns import PATTERNS


def get_case_variants(word: str) -> list[str]:
    """Get case variants of a word."""
    variants = [word]
    # Add capitalized version
    if word[0].islower():
        variants.append(word[0].upper() + word[1:])
    # Add lowercase version
    if word[0].isupper():
        variants.append(word[0].lower() + word[1:])
    return variants


def build_maps(tokenizer):
    """Build token mappings for pattern matching."""

    # Category word -> token IDs (with space and case variants)
    category_tokens = {}
    for cat, words in CATEGORIES.items():
        category_tokens[cat] = {}
        for word in words:
            all_tokens = []
            all_space_tokens = []

            # Add case variants
            for variant in get_case_variants(word):
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                space_tokens = tokenizer.encode(" " + variant, add_special_tokens=False)
                all_tokens.append(tokens)
                all_space_tokens.append(space_tokens)

            category_tokens[cat][word] = {
                "tokens": all_tokens[0],  # Primary
                "space_tokens": all_space_tokens[0],
                "all_tokens": all_tokens,  # All variants
                "all_space_tokens": all_space_tokens,
            }
            print(f"{cat}: {word!r} -> {all_tokens}")

    # Redirect phrase -> first token ID (for boosting)
    redirect_tokens = {}
    for pattern_name, pattern_data in PATTERNS.items():
        redirect_tokens[pattern_name] = []
        for phrase in pattern_data["redirects"]:
            if phrase:
                tokens = tokenizer.encode(phrase, add_special_tokens=False)
                space_tokens = tokenizer.encode(" " + phrase, add_special_tokens=False)

                redirect_tokens[pattern_name].append({
                    "phrase": phrase,
                    "tokens": tokens,
                    "space_tokens": space_tokens,
                    "first_token": tokens[0],
                    "space_first_token": space_tokens[0],
                })
                print(f"{pattern_name} redirect: {phrase!r} -> first={tokens[0]}")

    return {
        "category_tokens": category_tokens,
        "redirect_tokens": redirect_tokens,
    }


def main():
    print("Loading tokenizer...")
    _, tokenizer = load("mlx-community/Qwen3-4B-4bit")

    print("\nBuilding token maps...")
    maps = build_maps(tokenizer)

    output_path = Path(__file__).parent / "token_maps.json"
    with open(output_path, "w") as f:
        json.dump(maps, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Categories: {len(maps['category_tokens'])}")
    print(f"Patterns with redirects: {len(maps['redirect_tokens'])}")


if __name__ == "__main__":
    main()
