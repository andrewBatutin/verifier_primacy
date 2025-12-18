"""Base classes for verification rules.

All verification rules inherit from VerificationRule and implement
the get_allowed_mask method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import numpy as np

    Array = NDArray[np.floating]

from verifier_primacy.core.vocab import VocabAnalyzer


@dataclass
class ParserState:
    """State tracking for incremental JSON parsing.

    Attributes:
        in_string: Currently inside a string value
        in_object: Currently inside an object
        in_array: Currently inside an array
        depth: Nesting depth
        expected_type: Type expected for next value
        current_field: Name of current field being parsed
        buffer: Accumulated characters for current token
        position: Current position in output
    """

    in_string: bool = False
    in_object: bool = False
    in_array: bool = False
    depth: int = 0
    expected_type: str | None = None
    current_field: str | None = None
    buffer: str = ""
    position: int = 0

    def copy(self) -> "ParserState":
        """Create a copy of this state."""
        return ParserState(
            in_string=self.in_string,
            in_object=self.in_object,
            in_array=self.in_array,
            depth=self.depth,
            expected_type=self.expected_type,
            current_field=self.current_field,
            buffer=self.buffer,
            position=self.position,
        )


class VerificationRule(ABC):
    """Abstract base class for verification rules.

    Rules constrain which tokens are valid at each generation step.
    They examine the current parser state and return a mask indicating
    which tokens are allowed.

    To create a custom rule:
        1. Inherit from VerificationRule
        2. Implement get_allowed_mask()
        3. Optionally implement update_state() if you need to track
           additional state

    Example:
        >>> class NoSwearingRule(VerificationRule):
        ...     def __init__(self, vocab: VocabAnalyzer, bad_words: list[str]):
        ...         super().__init__(vocab)
        ...         self.bad_token_ids = self._find_bad_tokens(bad_words)
        ...
        ...     def get_allowed_mask(self, state: ParserState) -> Array:
        ...         mask = np.ones(self.vocab.vocab_size, dtype=np.float32)
        ...         mask[self.bad_token_ids] = 0.0
        ...         return mask
    """

    def __init__(self, vocab: VocabAnalyzer) -> None:
        """Initialize the rule with vocabulary analyzer.

        Args:
            vocab: VocabAnalyzer for the model's tokenizer
        """
        self.vocab = vocab

    @abstractmethod
    def get_allowed_mask(self, state: ParserState) -> "Array":
        """Return mask of allowed tokens given current state.

        Args:
            state: Current parser state

        Returns:
            Array of shape (vocab_size,) where:
                - 1.0 = token is allowed
                - 0.0 = token is forbidden
        """
        ...

    def update_state(self, state: ParserState, token_id: int) -> ParserState:
        """Update state after a token is sampled.

        Override this if your rule needs to track additional state
        beyond what's in ParserState.

        Args:
            state: Current parser state
            token_id: The token that was just sampled

        Returns:
            Updated parser state
        """
        return state

    def reset(self) -> None:
        """Reset any internal state.

        Called at the start of each generation.
        """
        pass


class CompositeRule(VerificationRule):
    """Combines multiple rules with AND logic.

    A token is only allowed if ALL rules allow it.
    """

    def __init__(self, vocab: VocabAnalyzer, rules: list[VerificationRule]) -> None:
        """Initialize with list of rules to combine.

        Args:
            vocab: VocabAnalyzer for the model's tokenizer
            rules: List of rules to combine
        """
        super().__init__(vocab)
        self.rules = rules

    def get_allowed_mask(self, state: ParserState) -> "Array":
        """Return intersection of all rule masks."""
        import numpy as np

        # Start with all allowed
        mask = np.ones(self.vocab.vocab_size, dtype=np.float32)

        # AND with each rule's mask
        for rule in self.rules:
            rule_mask = rule.get_allowed_mask(state)
            mask = np.minimum(mask, rule_mask)

        return mask

    def update_state(self, state: ParserState, token_id: int) -> ParserState:
        """Update state for all rules."""
        for rule in self.rules:
            state = rule.update_state(state, token_id)
        return state

    def reset(self) -> None:
        """Reset all rules."""
        for rule in self.rules:
            rule.reset()


class UnionRule(VerificationRule):
    """Combines multiple rules with OR logic.

    A token is allowed if ANY rule allows it.
    """

    def __init__(self, vocab: VocabAnalyzer, rules: list[VerificationRule]) -> None:
        """Initialize with list of rules to combine.

        Args:
            vocab: VocabAnalyzer for the model's tokenizer
            rules: List of rules to combine
        """
        super().__init__(vocab)
        self.rules = rules

    def get_allowed_mask(self, state: ParserState) -> "Array":
        """Return union of all rule masks."""
        import numpy as np

        # Start with none allowed
        mask = np.zeros(self.vocab.vocab_size, dtype=np.float32)

        # OR with each rule's mask
        for rule in self.rules:
            rule_mask = rule.get_allowed_mask(state)
            mask = np.maximum(mask, rule_mask)

        return mask

    def update_state(self, state: ParserState, token_id: int) -> ParserState:
        """Update state for all rules."""
        for rule in self.rules:
            state = rule.update_state(state, token_id)
        return state

    def reset(self) -> None:
        """Reset all rules."""
        for rule in self.rules:
            rule.reset()
