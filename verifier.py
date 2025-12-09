"""
Logits Verifier - Constrained decoding via rule-based logit masking.

Philosophy: Verify at generation time, don't pray and validate after.
The model's logits are a distribution - we constrain it to valid outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import mlx.core as mx
import numpy as np


class TokenClass(Enum):
    """Token classifications for rule matching."""
    WHITESPACE = auto()
    DIGIT = auto()
    ALPHA = auto()
    QUOTE = auto()
    COLON = auto()
    COMMA = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    DOT = auto()
    MINUS = auto()
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    NULL = auto()
    STRING_CONTENT = auto()  # any valid string character
    OTHER = auto()


@dataclass
class TokenInfo:
    """Precomputed token metadata for fast rule checking."""
    token_id: int
    text: str
    classes: set[TokenClass] = field(default_factory=set)
    
    def has_class(self, cls: TokenClass) -> bool:
        return cls in self.classes


class VocabAnalyzer:
    """
    Precompute token classifications for the entire vocabulary.
    Do this once at init, not per-token during generation.
    """

    def __init__(self, tokenizer, vocab_size: int | None = None):
        self.tokenizer = tokenizer
        # Use explicit vocab_size if provided, otherwise fall back to tokenizer
        # Note: tokenizer.vocab_size may be smaller than model's actual vocab
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        self.token_info: dict[int, TokenInfo] = {}
        self.class_masks: dict[TokenClass, mx.array] = {}

        self._analyze_vocab()
        self._build_class_masks()
    
    def _classify_token(self, text: str) -> set[TokenClass]:
        """Classify a single token's text."""
        classes = set()
        
        # Exact matches
        if text == "{":
            classes.add(TokenClass.LBRACE)
        elif text == "}":
            classes.add(TokenClass.RBRACE)
        elif text == "[":
            classes.add(TokenClass.LBRACKET)
        elif text == "]":
            classes.add(TokenClass.RBRACKET)
        elif text == ":":
            classes.add(TokenClass.COLON)
        elif text == ",":
            classes.add(TokenClass.COMMA)
        elif text == '"':
            classes.add(TokenClass.QUOTE)
        elif text == ".":
            classes.add(TokenClass.DOT)
        elif text == "-":
            classes.add(TokenClass.MINUS)
        elif text.lower() == "true":
            classes.add(TokenClass.BOOL_TRUE)
        elif text.lower() == "false":
            classes.add(TokenClass.BOOL_FALSE)
        elif text.lower() == "null":
            classes.add(TokenClass.NULL)
        
        # Pattern matches
        stripped = text.strip()
        if len(text) > 0 and len(stripped) == 0:
            # Only whitespace tokens (not tokens with leading/trailing whitespace)
            classes.add(TokenClass.WHITESPACE)
        if stripped.isdigit():
            classes.add(TokenClass.DIGIT)
        if stripped.isalpha():
            classes.add(TokenClass.ALPHA)
        
        # String content - anything that could appear inside quotes
        # (excluding unescaped quotes)
        if '"' not in text and '\\' not in text:
            classes.add(TokenClass.STRING_CONTENT)
        
        if not classes:
            classes.add(TokenClass.OTHER)
            
        return classes
    
    def _analyze_vocab(self):
        """Build token info for entire vocabulary."""
        for token_id in range(self.vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
                classes = self._classify_token(text)
                self.token_info[token_id] = TokenInfo(
                    token_id=token_id,
                    text=text,
                    classes=classes
                )
            except Exception:
                # Some token IDs may be invalid
                self.token_info[token_id] = TokenInfo(
                    token_id=token_id,
                    text="",
                    classes={TokenClass.OTHER}
                )
    
    def _build_class_masks(self):
        """Precompute boolean masks for each token class."""
        for cls in TokenClass:
            mask = np.zeros(self.vocab_size, dtype=np.float32)
            for token_id, info in self.token_info.items():
                if cls in info.classes:
                    mask[token_id] = 1.0
            self.class_masks[cls] = mx.array(mask)
    
    def get_class_mask(self, cls: TokenClass) -> mx.array:
        """Get precomputed mask for a token class (1.0 = allowed)."""
        return self.class_masks[cls]
    
    def get_combined_mask(self, classes: set[TokenClass]) -> mx.array:
        """Combine multiple class masks with OR."""
        if not classes:
            return mx.zeros(self.vocab_size)
        
        masks = [self.class_masks[cls] for cls in classes]
        combined = masks[0]
        for m in masks[1:]:
            combined = mx.maximum(combined, m)
        return combined


class VerificationRule(ABC):
    """Base class for logit verification rules."""
    
    @abstractmethod
    def get_allowed_mask(
        self, 
        state: "ParserState",
        vocab: VocabAnalyzer
    ) -> mx.array:
        """
        Return mask of allowed tokens (1.0 = allowed, 0.0 = forbidden).
        Will be converted to logit bias: 0.0 -> -inf
        """
        pass


@dataclass 
class ParserState:
    """
    Tracks parsing state for constrained generation.
    FSM state for JSON/parameter parsing.
    """
    # JSON structural state
    in_string: bool = False
    in_object: bool = False
    in_array: bool = False
    after_colon: bool = False
    after_comma: bool = False
    depth: int = 0
    
    # Schema state
    current_key: str | None = None
    expected_type: str | None = None
    
    # Buffer for partial tokens
    buffer: str = ""
    
    def copy(self) -> "ParserState":
        return ParserState(
            in_string=self.in_string,
            in_object=self.in_object,
            in_array=self.in_array,
            after_colon=self.after_colon,
            after_comma=self.after_comma,
            depth=self.depth,
            current_key=self.current_key,
            expected_type=self.expected_type,
            buffer=self.buffer
        )


class JSONStructureRule(VerificationRule):
    """
    Enforces valid JSON structure at the token level.
    
    State machine:
    - START -> expect { or [
    - OBJECT_START -> expect " (key) or }
    - OBJECT_KEY -> expect :
    - OBJECT_VALUE -> expect value then , or }
    - etc.
    """
    
    def get_allowed_mask(
        self, 
        state: ParserState,
        vocab: VocabAnalyzer
    ) -> mx.array:
        allowed_classes: set[TokenClass] = set()
        
        if state.in_string:
            # Inside string: allow string content or closing quote
            # Note: STRING_CONTENT is very broad, includes most tokens without " or \
            allowed_classes.add(TokenClass.STRING_CONTENT)
            allowed_classes.add(TokenClass.QUOTE)
            allowed_classes.add(TokenClass.ALPHA)
            allowed_classes.add(TokenClass.DIGIT)
            # Don't add WHITESPACE here - STRING_CONTENT already covers valid whitespace
            
        elif state.after_colon:
            # After colon: expect a value
            allowed_classes.add(TokenClass.QUOTE)  # string
            allowed_classes.add(TokenClass.DIGIT)  # number
            allowed_classes.add(TokenClass.MINUS)  # negative number
            allowed_classes.add(TokenClass.LBRACE)  # nested object
            allowed_classes.add(TokenClass.LBRACKET)  # array
            allowed_classes.add(TokenClass.BOOL_TRUE)
            allowed_classes.add(TokenClass.BOOL_FALSE)
            allowed_classes.add(TokenClass.NULL)
            allowed_classes.add(TokenClass.WHITESPACE)
            
        elif state.in_object and not state.after_comma:
            # In object, not after comma: expect key or close
            # Don't allow excessive whitespace - force model to make progress
            allowed_classes.add(TokenClass.QUOTE)  # key
            allowed_classes.add(TokenClass.RBRACE)  # close
            allowed_classes.add(TokenClass.COMMA)  # separator
            
        elif state.after_comma:
            # After comma in object: expect key (no whitespace to force progress)
            allowed_classes.add(TokenClass.QUOTE)
            
        else:
            # Default: allow structural tokens
            allowed_classes.add(TokenClass.LBRACE)
            allowed_classes.add(TokenClass.LBRACKET)
            allowed_classes.add(TokenClass.WHITESPACE)
        
        return vocab.get_combined_mask(allowed_classes)


@dataclass
class FieldSpec:
    """Specification for a parameter field."""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    required: bool = True
    enum: list[str] | None = None


class SchemaRule(VerificationRule):
    """
    Enforces parameter schema at generation time.
    Maps field names to expected types and constrains accordingly.
    """
    
    def __init__(self, fields: list[FieldSpec]):
        self.fields = {f.name: f for f in fields}
        self.field_names = set(f.name for f in fields)
    
    def get_allowed_mask(
        self, 
        state: ParserState,
        vocab: VocabAnalyzer
    ) -> mx.array:
        allowed_classes: set[TokenClass] = set()
        
        # If we know expected type, constrain to valid tokens
        if state.expected_type == "integer":
            allowed_classes.add(TokenClass.DIGIT)
            allowed_classes.add(TokenClass.MINUS)
        elif state.expected_type == "float":
            allowed_classes.add(TokenClass.DIGIT)
            allowed_classes.add(TokenClass.MINUS)
            allowed_classes.add(TokenClass.DOT)
        elif state.expected_type == "boolean":
            allowed_classes.add(TokenClass.BOOL_TRUE)
            allowed_classes.add(TokenClass.BOOL_FALSE)
        elif state.expected_type == "string":
            allowed_classes.add(TokenClass.QUOTE)
            allowed_classes.add(TokenClass.STRING_CONTENT)
            allowed_classes.add(TokenClass.ALPHA)
            allowed_classes.add(TokenClass.DIGIT)
        else:
            # No type constraint - allow all
            return mx.ones(vocab.vocab_size)
        
        # Always allow whitespace and structural tokens for flexibility
        allowed_classes.add(TokenClass.WHITESPACE)
        
        return vocab.get_combined_mask(allowed_classes)


class LogitsVerifier:
    """
    Main verifier that applies rules to logits during generation.
    
    Usage:
        verifier = LogitsVerifier(tokenizer, rules=[JSONStructureRule()])
        
        # In generation loop:
        logits = model(tokens)
        logits = verifier.apply(logits, state)
        next_token = sample(logits)
        state = verifier.update_state(state, next_token)
    """
    
    def __init__(
        self,
        tokenizer,
        rules: list[VerificationRule] | None = None,
        mask_value: float = -1e9,  # Large negative, not -inf for numerical stability
        vocab_size: int | None = None,  # Explicit vocab size (use model's embedding size)
    ):
        self.vocab = VocabAnalyzer(tokenizer, vocab_size=vocab_size)
        self.rules = rules or []
        self.mask_value = mask_value
        self.tokenizer = tokenizer
    
    def apply(
        self, 
        logits: mx.array, 
        state: ParserState
    ) -> mx.array:
        """
        Apply all rules to logits, masking invalid tokens.
        
        Args:
            logits: Raw logits from model [vocab_size] or [batch, vocab_size]
            state: Current parser state
            
        Returns:
            Masked logits with invalid tokens set to mask_value
        """
        if not self.rules:
            return logits
        
        # Combine masks from all rules (AND - token must be allowed by all)
        combined_mask = mx.ones(self.vocab.vocab_size)
        for rule in self.rules:
            rule_mask = rule.get_allowed_mask(state, self.vocab)
            combined_mask = mx.minimum(combined_mask, rule_mask)
        
        # Apply mask: where mask is 0, set logits to mask_value
        # mask: 1.0 = allowed, 0.0 = forbidden
        masked_logits = mx.where(
            combined_mask > 0.5,
            logits,
            mx.full(logits.shape, self.mask_value)
        )
        
        return masked_logits
    
    def update_state(
        self, 
        state: ParserState, 
        token_id: int
    ) -> ParserState:
        """
        Update parser state based on generated token.
        Returns new state (immutable update).
        """
        new_state = state.copy()
        token_text = self.tokenizer.decode([token_id])
        new_state.buffer += token_text
        
        # Update FSM state based on token
        for char in token_text:
            if char == '"':
                new_state.in_string = not new_state.in_string
            elif not new_state.in_string:
                if char == '{':
                    new_state.in_object = True
                    new_state.depth += 1
                elif char == '}':
                    new_state.depth -= 1
                    if new_state.depth == 0:
                        new_state.in_object = False
                elif char == '[':
                    new_state.in_array = True
                elif char == ']':
                    new_state.in_array = False
                elif char == ':':
                    new_state.after_colon = True
                elif char == ',':
                    new_state.after_colon = False
                    new_state.after_comma = True
                elif not char.isspace():
                    new_state.after_comma = False
        
        return new_state
    
    def create_initial_state(self) -> ParserState:
        """Create fresh parser state for new generation."""
        return ParserState()


def create_json_verifier(
    tokenizer,
    schema: list[FieldSpec] | None = None,
    vocab_size: int | None = None,
) -> LogitsVerifier:
    """
    Factory for JSON parameter parsing verifier.

    Args:
        tokenizer: HuggingFace/MLX tokenizer
        schema: Optional parameter schema for type constraints
        vocab_size: Explicit vocab size (use model.model.embed_tokens.weight.shape[0])
    """
    rules: list[VerificationRule] = [JSONStructureRule()]

    if schema:
        rules.append(SchemaRule(schema))

    return LogitsVerifier(tokenizer, rules=rules, vocab_size=vocab_size)
