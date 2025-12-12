"""Category-based pattern matching for slop detection."""

import json
from pathlib import Path

from categories import CATEGORIES
from patterns import PATTERNS


class PatternMatcher:
    """Detect slop patterns from token sequences."""

    def __init__(self, maps_path: str | Path | None = None):
        if maps_path is None:
            maps_path = Path(__file__).parent / "token_maps.json"

        with open(maps_path) as f:
            maps = json.load(f)

        self.category_tokens = maps["category_tokens"]
        self.redirect_tokens = maps["redirect_tokens"]

        # Build sequence map: tuple of tokens -> category
        # Only map single-token words to avoid false matches
        self.token_to_cat: dict[int, str] = {}
        self.sequence_to_cat: dict[tuple, str] = {}

        for cat, words in self.category_tokens.items():
            for word, token_data in words.items():
                # Get all variants (case + space)
                all_tokens = token_data.get("all_tokens", [token_data["tokens"]])
                all_space_tokens = token_data.get("all_space_tokens", [token_data["space_tokens"]])

                for tokens in all_tokens + all_space_tokens:
                    tokens = tuple(tokens)
                    # Single token: direct map
                    if len(tokens) == 1:
                        self.token_to_cat[tokens[0]] = cat
                    # Multi-token: sequence map
                    else:
                        self.sequence_to_cat[tokens] = cat

        # Build ban token sets per category (all variants)
        self.ban_tokens: dict[str, set[int]] = {}
        for cat, words in self.category_tokens.items():
            self.ban_tokens[cat] = set()
            for word, token_data in words.items():
                all_tokens = token_data.get("all_tokens", [token_data["tokens"]])
                all_space_tokens = token_data.get("all_space_tokens", [token_data["space_tokens"]])
                for tokens in all_tokens + all_space_tokens:
                    self.ban_tokens[cat].update(tokens)

    def get_recent_categories(self, generated_ids: list[int], window: int = 10) -> list[str]:
        """Convert recent token IDs to category sequence."""
        cats = []
        recent = generated_ids[-window:] if len(generated_ids) > window else generated_ids

        i = 0
        while i < len(recent):
            # Try multi-token sequences first (longest match)
            matched = False
            for seq_len in range(min(4, len(recent) - i), 0, -1):
                seq = tuple(recent[i:i + seq_len])
                if seq in self.sequence_to_cat:
                    cats.append(self.sequence_to_cat[seq])
                    i += seq_len
                    matched = True
                    break

            if not matched:
                # Try single token
                tid = recent[i]
                cat = self.token_to_cat.get(tid)
                if cat:
                    cats.append(cat)
                i += 1

        return cats

    def match_pattern(self, cats: list[str]) -> str | None:
        """Check if recent categories match any slop pattern."""
        if not cats:
            return None

        # REVEALER_SLOP: CONNECTOR? REVEALER -> about to say OBJECT
        if "REVEALER" in cats:
            return "REVEALER_SLOP"

        # HYPE_UNLOCK: HYPE ... UNLOCK or about to
        if "HYPE" in cats and "UNLOCK" in cats:
            return "HYPE_UNLOCK"

        # DEATH_CLAIM: * COPULA -> about to say DEATH
        if cats[-1] == "COPULA":
            return "DEATH_CLAIM"

        # FAKE_HUMILITY: detected FAKE_HUMBLE tokens
        if "FAKE_HUMBLE" in cats:
            return "FAKE_HUMILITY"

        # EXPLAINER_CRINGE: detected EXPLAINER tokens
        if "EXPLAINER" in cats:
            return "EXPLAINER_CRINGE"

        # STUDY_SLOP: detected STUDY_OPENER tokens
        if "STUDY_OPENER" in cats:
            return "STUDY_SLOP"

        return None

    def get_banned_tokens(self, pattern: str) -> set[int]:
        """Get tokens that continue the slop pattern."""
        ban_category = PATTERNS.get(pattern, {}).get("ban_category")
        if ban_category and ban_category in self.ban_tokens:
            return self.ban_tokens[ban_category]
        return set()

    def get_boost_tokens(self, pattern: str) -> list[int]:
        """Get tokens that start style alternatives."""
        if pattern not in self.redirect_tokens:
            return []

        tokens = []
        for redirect in self.redirect_tokens[pattern]:
            tokens.append(redirect["first_token"])
            tokens.append(redirect["space_first_token"])
        return tokens
