"""Tokenize banned phrases and save to JSON."""

import json
from pathlib import Path

from mlx_lm import load


def main():
    # Load tokenizer from model
    _, tokenizer = load("mlx-community/Qwen3-4B-4bit")

    # Read banned phrases
    banned_path = Path(__file__).parent / "banned.txt"
    banned_phrases = banned_path.read_text().strip().split("\n")

    # Tokenize each phrase (with and without leading space)
    banned_token_seqs = {}
    for phrase in banned_phrases:
        # Original phrase
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        banned_token_seqs[phrase] = tokens
        print(f"{phrase!r} -> {tokens}")

        # Space-prefixed variant (catches mid-sentence usage)
        space_phrase = " " + phrase
        space_tokens = tokenizer.encode(space_phrase, add_special_tokens=False)
        if space_tokens != tokens:  # Only add if different
            banned_token_seqs[space_phrase] = space_tokens
            print(f"{space_phrase!r} -> {space_tokens}")

    # Save to JSON
    output_path = Path(__file__).parent / "banned_tokens.json"
    with open(output_path, "w") as f:
        json.dump(banned_token_seqs, f, indent=2)

    print(f"\nSaved {len(banned_token_seqs)} banned sequences to {output_path}")


if __name__ == "__main__":
    main()
