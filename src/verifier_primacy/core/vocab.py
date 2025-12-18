"""Vocabulary analysis for token classification.

This module provides VocabAnalyzer which precomputes token classifications
and masks for efficient constrained decoding.
"""

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[np.floating]


class TokenClass(Flag):
    """Classification flags for tokens.

    Tokens can belong to multiple classes (e.g., DIGIT | JSON_VALUE).
    """

    NONE = 0

    # Character types
    DIGIT = auto()
    ALPHA = auto()
    WHITESPACE = auto()
    PUNCTUATION = auto()

    # JSON structural
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COLON = auto()  # :
    COMMA = auto()  # ,
    QUOTE = auto()  # "

    # JSON values
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    NULL = auto()

    # Composite classes
    JSON_VALUE = DIGIT | BOOL_TRUE | BOOL_FALSE | NULL | QUOTE
    JSON_STRUCTURAL = LBRACE | RBRACE | LBRACKET | RBRACKET | COLON | COMMA


class Tokenizer(Protocol):
    """Protocol for tokenizers compatible with VocabAnalyzer."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to string."""
        ...

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        ...


@dataclass
class VocabAnalyzer:
    """Precomputes token classifications for efficient constrained decoding.

    Analyzes the tokenizer's vocabulary once at initialization, creating
    boolean masks for each token class. During generation, these masks
    can be combined with simple boolean operations.

    Attributes:
        tokenizer: The tokenizer to analyze
        vocab_size: Size of the vocabulary
        masks: Dict mapping TokenClass to boolean masks

    Example:
        >>> vocab = VocabAnalyzer(tokenizer)
        >>> digit_mask = vocab.get_mask(TokenClass.DIGIT)
        >>> # During generation:
        >>> allowed = digit_mask | vocab.get_mask(TokenClass.WHITESPACE)
    """

    tokenizer: Tokenizer
    vocab_size: int = field(init=False)
    masks: dict[TokenClass, "Array"] = field(init=False, default_factory=dict)
    _token_classes: list[TokenClass] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Analyze vocabulary and build masks."""
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._analyze_vocab()

    def _analyze_vocab(self) -> None:
        """Classify all tokens in the vocabulary."""
        self._token_classes = [TokenClass.NONE] * self.vocab_size

        for token_id in range(self.vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id])
                self._token_classes[token_id] = self._classify_token(token_str)
            except Exception:
                # Some token IDs may be invalid
                self._token_classes[token_id] = TokenClass.NONE

        # Build masks for each class
        self._build_masks()

    def _classify_token(self, token_str: str) -> TokenClass:
        """Classify a single token string."""
        if not token_str:
            return TokenClass.NONE

        result = TokenClass.NONE

        # Check character types
        if token_str.isdigit() or (
            token_str.startswith("-") and token_str[1:].isdigit()
        ):
            result |= TokenClass.DIGIT
        if token_str.isalpha():
            result |= TokenClass.ALPHA
        if token_str.isspace():
            result |= TokenClass.WHITESPACE

        # Check JSON structural
        token_stripped = token_str.strip()
        if token_stripped == "{":
            result |= TokenClass.LBRACE
        elif token_stripped == "}":
            result |= TokenClass.RBRACE
        elif token_stripped == "[":
            result |= TokenClass.LBRACKET
        elif token_stripped == "]":
            result |= TokenClass.RBRACKET
        elif token_stripped == ":":
            result |= TokenClass.COLON
        elif token_stripped == ",":
            result |= TokenClass.COMMA
        elif token_stripped == '"':
            result |= TokenClass.QUOTE

        # Check JSON values
        if token_str.lower() == "true":
            result |= TokenClass.BOOL_TRUE
        elif token_str.lower() == "false":
            result |= TokenClass.BOOL_FALSE
        elif token_str.lower() == "null":
            result |= TokenClass.NULL

        return result

    def _build_masks(self) -> None:
        """Build boolean masks for each token class."""
        # Get all individual token classes (not composite)
        individual_classes = [
            tc
            for tc in TokenClass
            if tc != TokenClass.NONE
            and tc not in (TokenClass.JSON_VALUE, TokenClass.JSON_STRUCTURAL)
        ]

        for token_class in individual_classes:
            mask = np.zeros(self.vocab_size, dtype=np.float32)
            for token_id, tc in enumerate(self._token_classes):
                if token_class in tc:
                    mask[token_id] = 1.0
            self.masks[token_class] = mask

    def get_mask(self, token_class: TokenClass) -> "Array":
        """Get the boolean mask for a token class.

        Args:
            token_class: The class or combination of classes

        Returns:
            Boolean mask of shape (vocab_size,) where 1.0 = allowed
        """
        # Handle composite classes
        if token_class == TokenClass.JSON_VALUE:
            return self._combine_masks(
                [TokenClass.DIGIT, TokenClass.BOOL_TRUE, TokenClass.BOOL_FALSE, TokenClass.NULL, TokenClass.QUOTE]
            )
        elif token_class == TokenClass.JSON_STRUCTURAL:
            return self._combine_masks(
                [
                    TokenClass.LBRACE,
                    TokenClass.RBRACE,
                    TokenClass.LBRACKET,
                    TokenClass.RBRACKET,
                    TokenClass.COLON,
                    TokenClass.COMMA,
                ]
            )

        if token_class in self.masks:
            return self.masks[token_class]

        # Handle combinations
        return self._combine_masks(
            [tc for tc in TokenClass if tc in token_class and tc in self.masks]
        )

    def _combine_masks(self, classes: list[TokenClass]) -> "Array":
        """Combine multiple token class masks with OR logic."""
        if not classes:
            return np.zeros(self.vocab_size, dtype=np.float32)

        result = np.zeros(self.vocab_size, dtype=np.float32)
        for tc in classes:
            if tc in self.masks:
                result = np.maximum(result, self.masks[tc])

        return result

    def get_class(self, token_id: int) -> TokenClass:
        """Get the class(es) for a specific token ID.

        Args:
            token_id: The token ID to look up

        Returns:
            TokenClass flags for this token
        """
        if 0 <= token_id < self.vocab_size:
            return self._token_classes[token_id]
        return TokenClass.NONE

    def get_tokens_by_class(self, token_class: TokenClass) -> list[int]:
        """Get all token IDs that belong to a class.

        Args:
            token_class: The class to filter by

        Returns:
            List of token IDs in this class
        """
        mask = self.get_mask(token_class)
        return [i for i, m in enumerate(mask) if m > 0]
