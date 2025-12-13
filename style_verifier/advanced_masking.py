"""Advanced masking techniques for style verification."""

import json
from pathlib import Path

import mlx.core as mx

from style_verifier import StyleVerifier


# Slop patterns for embedding detection
SLOP_PHRASES = {
    "revealer": ["And here's what", "Here's the thing", "Here is what"],
    "unlock": ["The real unlock is", "The real secret is", "The actual secret"],
    "death": ["is dead", "are dead", "is dying"],
    "study": ["A study of", "A recent study", "Research shows"],
    "humble": ["I was wrong", "I used to think", "I admit"],
    "explainer": ["Let me explain", "In this article", "In this thread"],
}

# Alternative phrases to boost
ALTERNATIVES = {
    "revealer": ["The data shows", "Measured:", "Evidence:", "Three factors:"],
    "unlock": ["The mechanism:", "How it works:", "Components:", "The constraint:"],
    "death": ["is declining", "dropped to", "decreased by"],
    "study": ["In my opinion", "Based on experience", "From observation"],
    "humble": ["Previously:", "Old approach:", "Before:"],
    "explainer": ["Summary:", "Key points:", "Overview:"],
}


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two vectors."""
    dot = mx.sum(a * b)
    norm_a = mx.sqrt(mx.sum(a * a))
    norm_b = mx.sqrt(mx.sum(b * b))
    return (dot / (norm_a * norm_b + 1e-10)).item()


def get_embedding(model, tokens: list[int]) -> mx.array:
    """Get mean embedding for token sequence."""
    if not tokens:
        return None
    token_array = mx.array(tokens)
    embeddings = model.model.embed_tokens(token_array)
    return mx.mean(embeddings, axis=0)


def compute_entropy(logits: mx.array) -> float:
    """Compute entropy of probability distribution."""
    probs = mx.softmax(logits)
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    return entropy.item()


class AdvancedStyleProcessor:
    """Combined processor with toggleable masking techniques."""

    def __init__(
        self,
        model,
        tokenizer,
        config: dict,
        verifier: StyleVerifier = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.verifier = verifier or StyleVerifier()

        # Config options with defaults
        self.use_hard_mask = config.get("hard_mask", True)
        self.use_soft_mask = config.get("soft_mask", False)
        self.use_embedding = config.get("embedding", False)
        self.use_entropy = config.get("entropy_adaptive", False)
        self.use_lookahead = config.get("lookahead", False)
        self.use_boost = config.get("boost_alternatives", False)

        # Parameters
        self.soft_penalty = config.get("soft_penalty", -10.0)
        self.embedding_threshold = config.get("embedding_threshold", 0.80)
        self.entropy_threshold = config.get("entropy_threshold", 2.0)
        self.lookahead_depth = config.get("lookahead_depth", 3)
        self.boost_strength = config.get("boost_strength", 5.0)

        # Pre-compute slop embeddings if embedding detection enabled
        self.slop_embeddings = {}
        if self.use_embedding:
            self._compute_slop_embeddings()

        # Pre-compute alternative token IDs if boosting enabled
        self.alternative_tokens = {}
        if self.use_boost:
            self._compute_alternative_tokens()

        # Metrics for debugging
        self.last_metrics = {}

    def _compute_slop_embeddings(self):
        """Pre-compute embeddings for slop patterns."""
        for pattern_name, phrases in SLOP_PHRASES.items():
            embeddings = []
            for phrase in phrases:
                tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
                emb = get_embedding(self.model, tokens)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                # Average all phrase embeddings for this pattern
                self.slop_embeddings[pattern_name] = mx.mean(mx.stack(embeddings), axis=0)

    def _compute_alternative_tokens(self):
        """Pre-compute first tokens of alternative phrases."""
        for pattern_name, phrases in ALTERNATIVES.items():
            tokens = set()
            for phrase in phrases:
                toks = self.tokenizer.encode(phrase, add_special_tokens=False)
                if toks:
                    tokens.add(toks[0])
                # Also add space-prefixed version
                space_toks = self.tokenizer.encode(" " + phrase, add_special_tokens=False)
                if space_toks:
                    tokens.add(space_toks[0])
            self.alternative_tokens[pattern_name] = list(tokens)

    def detect_slop_by_embedding(self, context: list[int]) -> str | None:
        """Detect if recent context is heading toward slop pattern."""
        if not context or not self.slop_embeddings:
            return None

        # Get embedding of recent tokens
        recent = context[-10:] if len(context) > 10 else context
        recent_emb = get_embedding(self.model, recent)
        if recent_emb is None:
            return None

        # Check similarity to each slop pattern
        best_match = None
        best_sim = 0.0

        for pattern_name, pattern_emb in self.slop_embeddings.items():
            sim = cosine_similarity(recent_emb, pattern_emb)
            if sim > self.embedding_threshold and sim > best_sim:
                best_match = pattern_name
                best_sim = sim

        self.last_metrics["embedding_sim"] = best_sim
        self.last_metrics["detected_pattern"] = best_match
        return best_match

    def lookahead_detect_slop(self, tokens: mx.array, logits: mx.array) -> bool:
        """Simulate greedy generation to check if path leads to slop."""
        simulated = tokens.tolist()

        for _ in range(self.lookahead_depth):
            next_token = mx.argmax(logits).item()
            simulated.append(next_token)

            # Check if simulated path contains slop
            text = self.tokenizer.decode(simulated[-10:])
            for phrases in SLOP_PHRASES.values():
                for phrase in phrases:
                    if phrase.lower() in text.lower():
                        return True

            # Get next logits (expensive!)
            logits = self.model(mx.array([simulated]))[0, -1, :]

        return False

    def apply_soft_mask(self, logits: mx.array, context: list[int]) -> mx.array:
        """Apply graduated penalties instead of hard -inf mask."""
        tokens_to_penalize = []

        # Get tokens that would be hard-masked
        for seq in self.verifier.banned_token_lists:
            for i in range(1, len(seq)):
                prefix = seq[:i]
                next_token = seq[i]
                if len(context) >= len(prefix):
                    if context[-len(prefix):] == prefix:
                        tokens_to_penalize.append(next_token)

        if not tokens_to_penalize:
            return logits

        # Apply soft penalty
        vocab_size = logits.shape[-1]
        penalty_mask = mx.zeros(vocab_size)
        for tid in tokens_to_penalize:
            if tid < vocab_size:
                penalty_mask = penalty_mask.at[tid].add(self.soft_penalty)

        return logits + penalty_mask

    def apply_boost(self, logits: mx.array, detected_pattern: str) -> mx.array:
        """Boost alternative tokens for detected pattern."""
        if detected_pattern is None or detected_pattern not in self.alternative_tokens:
            return logits

        vocab_size = logits.shape[-1]
        boost_mask = mx.zeros(vocab_size)

        for tid in self.alternative_tokens[detected_pattern]:
            if tid < vocab_size:
                boost_mask = boost_mask.at[tid].add(self.boost_strength)

        return logits + boost_mask

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply all enabled masking techniques."""
        context = tokens.tolist()
        original_logits = logits

        # Reset metrics
        self.last_metrics = {
            "entropy": compute_entropy(logits),
            "embedding_sim": 0.0,
            "detected_pattern": None,
            "techniques_applied": [],
        }

        # 1. Entropy check - skip intervention if model is confident
        if self.use_entropy:
            entropy = self.last_metrics["entropy"]
            if entropy < self.entropy_threshold:
                self.last_metrics["techniques_applied"].append("entropy_skip")
                return logits  # Model confident, don't intervene

        # 2. Embedding-based detection
        detected_pattern = None
        if self.use_embedding:
            detected_pattern = self.detect_slop_by_embedding(context)
            if detected_pattern:
                self.last_metrics["techniques_applied"].append(f"embedding:{detected_pattern}")

        # 3. Lookahead detection (expensive - use sparingly)
        if self.use_lookahead and not detected_pattern:
            if self.lookahead_detect_slop(tokens, logits):
                detected_pattern = "lookahead"
                self.last_metrics["techniques_applied"].append("lookahead")

        # 4. Apply masking (hard or soft)
        if self.use_hard_mask:
            logits = self.verifier.mask(logits, context)
            self.last_metrics["techniques_applied"].append("hard_mask")
        elif self.use_soft_mask:
            logits = self.apply_soft_mask(logits, context)
            self.last_metrics["techniques_applied"].append("soft_mask")

        # 5. Boost alternatives
        if self.use_boost and detected_pattern:
            logits = self.apply_boost(logits, detected_pattern)
            self.last_metrics["techniques_applied"].append(f"boost:{detected_pattern}")

        return logits


def make_advanced_processor(model, tokenizer, config: dict, verifier: StyleVerifier = None):
    """Create a logits processor with advanced masking."""
    processor = AdvancedStyleProcessor(model, tokenizer, config, verifier)

    def process(tokens: mx.array, logits: mx.array) -> mx.array:
        return processor(tokens, logits)

    # Attach processor for metrics access
    process.processor = processor
    return process
