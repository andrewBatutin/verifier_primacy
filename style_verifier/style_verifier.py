"""Style verifier that masks logits for banned phrases."""

import json
from pathlib import Path

import mlx.core as mx


class StyleVerifier:
    """Masks logits to prevent generation of banned token sequences."""

    def __init__(self, banned_path: str | Path | None = None):
        if banned_path is None:
            banned_path = Path(__file__).parent / "banned_tokens.json"

        with open(banned_path) as f:
            self.banned_seqs = json.load(f)

        self.banned_token_lists = list(self.banned_seqs.values())

    def mask(self, logits: mx.array, generated: list[int]) -> mx.array:
        """Mask logits to prevent banned sequences.

        For each banned sequence, if the recently generated tokens match
        a prefix of that sequence, set the logit of the next token in
        the sequence to -inf.

        Args:
            logits: Logits array of shape (vocab_size,)
            generated: List of token IDs generated so far

        Returns:
            Modified logits with banned continuations masked
        """
        vocab_size = logits.shape[-1]
        tokens_to_mask = []

        for seq in self.banned_token_lists:
            for i in range(1, len(seq)):
                prefix = seq[:i]
                next_token = seq[i]

                if len(generated) >= len(prefix):
                    if generated[-len(prefix) :] == prefix:
                        if next_token < vocab_size:
                            tokens_to_mask.append(next_token)

        if not tokens_to_mask:
            return logits

        # Create mask: 0 for tokens to mask, 1 otherwise
        mask = mx.ones(vocab_size)
        for token_id in tokens_to_mask:
            mask = mask.at[token_id].add(-1)

        # Apply mask: -inf for masked tokens
        neg_inf = mx.array(float("-inf"))
        return mx.where(mask > 0, logits, neg_inf)
