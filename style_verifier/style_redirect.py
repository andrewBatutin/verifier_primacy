"""Style redirect logit processor - mask slop + boost alternatives."""

import json
from pathlib import Path

import mlx.core as mx

from pattern_matcher import PatternMatcher


class StyleRedirect:
    """Logit processor that masks slop continuations and boosts style alternatives."""

    def __init__(self, maps_path: str | Path | None = None, boost: float = 5.0):
        self.matcher = PatternMatcher(maps_path)
        self.boost = boost

        # Load full redirect sequences for continuation boosting
        if maps_path is None:
            maps_path = Path(__file__).parent / "token_maps.json"
        with open(maps_path) as f:
            maps = json.load(f)
        self.redirect_tokens = maps.get("redirect_tokens", {})

    def _get_continuation_token(self, generated_ids: list[int], pattern: str) -> int | None:
        """Check if we're mid-redirect and return next token to boost."""
        if pattern not in self.redirect_tokens:
            return None

        for redirect in self.redirect_tokens[pattern]:
            tokens = redirect["tokens"]
            space_tokens = redirect["space_tokens"]

            # Check both variants
            for seq in [tokens, space_tokens]:
                if len(seq) <= 1:
                    continue
                # Check if generated ends with a prefix of this redirect
                for prefix_len in range(1, len(seq)):
                    prefix = seq[:prefix_len]
                    if len(generated_ids) >= prefix_len:
                        if generated_ids[-prefix_len:] == prefix:
                            # Return next token in sequence
                            return seq[prefix_len]
        return None

    def _is_in_redirect_sequence(self, generated_ids: list[int], pattern: str) -> bool:
        """Check if we're currently generating a redirect sequence."""
        if pattern not in self.redirect_tokens:
            return False

        for redirect in self.redirect_tokens[pattern]:
            tokens = redirect["tokens"]
            space_tokens = redirect["space_tokens"]

            for seq in [tokens, space_tokens]:
                # Check if generated ends with any prefix of this redirect
                for prefix_len in range(1, len(seq)):
                    prefix = seq[:prefix_len]
                    if len(generated_ids) >= prefix_len:
                        if generated_ids[-prefix_len:] == prefix:
                            return True
        return False

    def __call__(self, logits: mx.array, generated_ids: list[int]) -> mx.array:
        """Apply style redirect to logits.

        Args:
            logits: Logits array of shape (vocab_size,)
            generated_ids: List of token IDs generated so far

        Returns:
            Modified logits with alternatives boosted (no banning - use StyleVerifier for that)
        """
        vocab_size = logits.shape[-1]
        boost_tokens = []

        # Early boost: at very start of generation,
        # boost STUDY_SLOP redirects to encourage "In my humble opinion" style
        # Only boost if we're in the first few tokens or continuing a redirect
        if len(generated_ids) <= 1:
            boost_tokens = self.matcher.get_boost_tokens("STUDY_SLOP")
        elif len(generated_ids) <= 10:
            # Check if we're continuing a redirect sequence
            cont_token = self._get_continuation_token(generated_ids, "STUDY_SLOP")
            if cont_token is not None:
                boost_tokens.append(cont_token)

        # Skip pattern-based boosting - it causes loops with common tokens
        # Pattern detection is better suited for banning (StyleVerifier) than boosting

        if not boost_tokens:
            return logits

        # Apply boost to alternatives
        boost_mask = mx.zeros(vocab_size)
        for tid in boost_tokens:
            if tid < vocab_size:
                boost_mask = boost_mask.at[tid].add(self.boost)
        logits = logits + boost_mask

        return logits


def make_redirect_processor(boost: float = 5.0):
    """Create a logits processor for mlx_lm.generate().

    Args:
        boost: Amount to boost style alternative tokens

    Returns:
        A function compatible with mlx_lm's logits_processors parameter
    """
    redirector = StyleRedirect(boost=boost)

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        generated = tokens.tolist()
        return redirector(logits, generated)

    return processor
