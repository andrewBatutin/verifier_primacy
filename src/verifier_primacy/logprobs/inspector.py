"""TokenInspector - Utilities for understanding tokenization.

This module provides tools to inspect how text is tokenized,
search vocabularies, and visualize token boundaries.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class TokenInfo(BaseModel):
    """Information about a single token.

    Attributes:
        token: The decoded token text.
        token_id: The token's ID in the vocabulary.
        byte_length: Number of bytes in the token.
        is_special: Whether this is a special token (BOS, EOS, etc.).
    """

    token: str = Field(description="Decoded token text")
    token_id: int = Field(description="Token ID in vocabulary")
    byte_length: int = Field(description="Number of bytes")
    is_special: bool = Field(default=False, description="Is a special token")

    def __str__(self) -> str:
        special = " [SPECIAL]" if self.is_special else ""
        return f"[{self.token_id}] '{self.token}' ({self.byte_length}b){special}"


class TokenInspection(BaseModel):
    """Result of inspecting text tokenization.

    Attributes:
        text: The original input text.
        tokens: List of token information.
        token_count: Total number of tokens.
    """

    text: str = Field(description="Original input text")
    tokens: list[TokenInfo] = Field(default_factory=list, description="Token information")
    token_count: int = Field(description="Number of tokens")

    def __str__(self) -> str:
        """Human-readable token breakdown."""
        token_strs = [f"[{t.token}]" for t in self.tokens]
        return "".join(token_strs)

    def visualize(self, separator: str = "|") -> str:
        """Visualize token boundaries with separators.

        Args:
            separator: Character to use between tokens.

        Returns:
            Text with token boundaries marked.
        """
        parts = [t.token for t in self.tokens]
        return separator.join(parts)

    def to_table(self) -> str:
        """Format as a table showing token details.

        Returns:
            Multi-line string with token table.
        """
        lines = ["Pos | ID      | Token          | Bytes"]
        lines.append("-" * 45)
        for i, t in enumerate(self.tokens):
            token_repr = repr(t.token)
            if len(token_repr) > 14:
                token_repr = token_repr[:11] + "..."
            lines.append(f"{i:3} | {t.token_id:7} | {token_repr:14} | {t.byte_length}")
        return "\n".join(lines)


class TokenInspector:
    """Inspect tokenization and search vocabularies.

    Helps understand how text is broken into tokens,
    which is essential for understanding logprobs.

    Example:
        >>> inspector = TokenInspector(explorer.backend.tokenizer)
        >>> inspection = inspector.inspect("Hello, world!")
        >>> print(inspection.visualize())
        Hello|,| world|!
        >>> print(inspection.to_table())
        Pos | ID      | Token          | Bytes
        ---------------------------------------------
          0 |   15496 | 'Hello'        | 5
          1 |     11  | ','            | 1
        ...
    """

    def __init__(self, tokenizer: Any) -> None:
        """Initialize the inspector.

        Args:
            tokenizer: A tokenizer with encode/decode methods.
        """
        self._tokenizer = tokenizer
        self._special_tokens: set[int] = set()
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize set of special token IDs."""
        # Try to get special tokens from tokenizer
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            token_id = getattr(self._tokenizer, attr, None)
            if token_id is not None:
                self._special_tokens.add(token_id)

        # Try to get all special tokens
        if hasattr(self._tokenizer, "all_special_ids"):
            self._special_tokens.update(self._tokenizer.all_special_ids)

    def inspect(self, text: str) -> TokenInspection:
        """Inspect how text is tokenized.

        Args:
            text: The text to tokenize and inspect.

        Returns:
            TokenInspection with detailed token information.
        """
        token_ids = self._tokenizer.encode(text)
        tokens = []

        for token_id in token_ids:
            decoded = self._decode_single(token_id)
            tokens.append(
                TokenInfo(
                    token=decoded,
                    token_id=token_id,
                    byte_length=len(decoded.encode("utf-8")),
                    is_special=token_id in self._special_tokens,
                )
            )

        return TokenInspection(
            text=text,
            tokens=tokens,
            token_count=len(tokens),
        )

    def _decode_single(self, token_id: int) -> str:
        """Decode a single token ID.

        Args:
            token_id: The token ID to decode.

        Returns:
            The decoded token text.
        """
        # Try direct decode
        try:
            return self._tokenizer.decode([token_id])
        except Exception:
            # Fallback: try to get from vocab
            if hasattr(self._tokenizer, "convert_ids_to_tokens"):
                return self._tokenizer.convert_ids_to_tokens([token_id])[0]
            return f"<{token_id}>"

    def find_token(self, query: str) -> list[TokenInfo]:
        """Find tokens containing a string.

        Args:
            query: String to search for in tokens.

        Returns:
            List of matching tokens.
        """
        results = []
        vocab_size = self._get_vocab_size()

        for token_id in range(min(vocab_size, 100000)):  # Cap for performance
            try:
                decoded = self._decode_single(token_id)
                if query in decoded:
                    results.append(
                        TokenInfo(
                            token=decoded,
                            token_id=token_id,
                            byte_length=len(decoded.encode("utf-8")),
                            is_special=token_id in self._special_tokens,
                        )
                    )
                    if len(results) >= 50:  # Limit results
                        break
            except Exception:
                continue

        return results

    def vocab_search(self, pattern: str, limit: int = 50) -> list[TokenInfo]:
        """Search vocabulary with a regex pattern.

        Args:
            pattern: Regular expression to match tokens.
            limit: Maximum results to return.

        Returns:
            List of matching tokens.
        """
        regex = re.compile(pattern)
        results = []
        vocab_size = self._get_vocab_size()

        for token_id in range(min(vocab_size, 100000)):
            try:
                decoded = self._decode_single(token_id)
                if regex.search(decoded):
                    results.append(
                        TokenInfo(
                            token=decoded,
                            token_id=token_id,
                            byte_length=len(decoded.encode("utf-8")),
                            is_special=token_id in self._special_tokens,
                        )
                    )
                    if len(results) >= limit:
                        break
            except Exception:
                continue

        return results

    def _get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        if hasattr(self._tokenizer, "__len__"):
            return len(self._tokenizer)
        return 50000  # Default fallback

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        return len(self._tokenizer.encode(text))

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count without full tokenization.

        Uses a rough heuristic (4 chars per token on average).
        For exact counts, use count_tokens().

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        # Rough heuristic: ~4 characters per token for English
        return max(1, len(text) // 4)
