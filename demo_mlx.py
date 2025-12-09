#!/usr/bin/env python3
"""
Demo: Constrained JSON generation with MLX.

This shows the core loop:
1. Get logits from model
2. Apply verification rules (mask invalid tokens)
3. Sample from constrained distribution
4. Update parser state
5. Repeat

The beauty: model can't generate invalid JSON. Period.
No regex validation, no retry loops, no prayer.
"""

import mlx.core as mx
from mlx_lm import load

from verifier import (
    LogitsVerifier,
    JSONStructureRule,
    SchemaRule,
    FieldSpec,
    ParserState,
    create_json_verifier,
)


def constrained_generate(
    model,
    tokenizer,
    verifier: LogitsVerifier,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    start_in_string: bool = False,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate with logits-level constraints.

    Core insight: We intercept logits BEFORE sampling.
    Invalid tokens get -inf, can't be sampled.
    """
    # Tokenize prompt
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    # Initialize state
    state = verifier.create_initial_state()
    if start_in_string:
        # If prompt ends inside a string (after "), set state accordingly
        state.in_object = True
        state.in_string = True
        state.after_colon = True  # We're after {"action":
        state.depth = 1
    generated_tokens = []
    token_counts: dict[int, int] = {}  # Track token frequencies for repetition penalty
    
    # Generation loop
    cache = None
    for _ in range(max_tokens):
        # Forward pass
        if cache is None:
            logits = model(prompt_tokens[None, :])
            # Get last token's logits
            logits = logits[:, -1, :]
        else:
            last_token = mx.array([[generated_tokens[-1]]])
            logits = model(last_token, cache=cache)
            logits = logits[:, -1, :]
        
        # THE KEY STEP: Apply verification rules
        logits = logits.squeeze(0)  # [vocab_size]
        constrained_logits = verifier.apply(logits, state)

        # Apply repetition penalty to avoid loops
        if repetition_penalty != 1.0:
            for tid, count in token_counts.items():
                if constrained_logits[tid] > 0:
                    constrained_logits = constrained_logits.at[tid].divide(repetition_penalty ** count)
                else:
                    constrained_logits = constrained_logits.at[tid].multiply(repetition_penalty ** count)

        # Sample from constrained distribution
        # Note: mx.random.categorical expects logits, not probabilities
        if temperature > 0:
            token_id = mx.random.categorical(constrained_logits / temperature).item()
        else:
            token_id = mx.argmax(constrained_logits).item()

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(token_id)
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        # Update parser state
        state = verifier.update_state(state, token_id)
        
        # Check if JSON is complete (depth back to 0 after starting)
        if state.depth == 0 and len(generated_tokens) > 1:
            break
    
    return tokenizer.decode(generated_tokens)


def demo_parameter_extraction():
    """
    Demo: Show how the logits verifier constrains generation.

    This demonstrates:
    1. Token classification at init time
    2. Logit masking based on JSON grammar state
    3. How invalid tokens get -inf and can't be sampled
    """
    print("=" * 60)
    print("LOGITS VERIFIER DEMO")
    print("Constrained JSON generation - verify don't pray")
    print("=" * 60)

    # Load model (Qwen3-4B is much better at JSON generation)
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    vocab_size = model.model.embed_tokens.weight.shape[0]

    # Create verifier
    schema = [
        FieldSpec(name="action", type="string", enum=["search", "create", "delete"]),
    ]
    verifier = create_json_verifier(tokenizer, schema, vocab_size=vocab_size)

    print(f"\nVocab size: {vocab_size}")
    print(f"Verifier rules: {len(verifier.rules)}")

    # Demo 1: Show token masking at start of JSON
    print("\n" + "-" * 60)
    print("DEMO 1: Token masking at JSON start")
    print("-" * 60)

    state = verifier.create_initial_state()
    mask = verifier.rules[0].get_allowed_mask(state, verifier.vocab)
    allowed = int(mx.sum(mask).item())

    print(f"At JSON start, only {allowed} tokens allowed out of {vocab_size}")
    print("Allowed tokens (sample): {, [, and whitespace variants")

    # Demo 2: Show masking inside an object
    print("\n" + "-" * 60)
    print("DEMO 2: Token masking inside object (after {)")
    print("-" * 60)

    state = ParserState(in_object=True, depth=1)
    mask = verifier.rules[0].get_allowed_mask(state, verifier.vocab)
    allowed = int(mx.sum(mask).item())

    print(f"Inside object, only {allowed} tokens allowed")
    print('Allowed: " (for keys), } (to close), , (separator)')

    # Demo 3: Actual generation with masking
    print("\n" + "-" * 60)
    print("DEMO 3: Constrained generation")
    print("-" * 60)

    prompt = 'Output valid JSON: {"name": "'
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    state = ParserState(in_object=True, in_string=True, depth=1)

    logits = model(prompt_tokens[None, :])[:, -1, :].squeeze(0)
    constrained = verifier.apply(logits, state)

    # Show stats
    allowed_count = int(mx.sum(constrained > -1e8).item())
    print(f"After masking: {allowed_count} tokens allowed (inside string)")

    # Show top predictions
    top_ids = mx.argsort(constrained)[-5:][::-1].tolist()
    print("\nTop 5 constrained predictions:")
    for tid in top_ids:
        txt = tokenizer.decode([tid])
        print(f"  {repr(txt):20} (logit: {constrained[tid].item():.1f})")

    # Demo 4: Show that invalid tokens really are blocked
    print("\n" + "-" * 60)
    print("DEMO 4: Verification that invalid tokens are blocked")
    print("-" * 60)

    # At JSON start, these should be blocked
    state = verifier.create_initial_state()
    test_tokens = [
        (tokenizer.encode("hello")[0], "hello"),
        (tokenizer.encode("123")[0], "123"),
        (tokenizer.encode("{")[0], "{"),
    ]

    dummy_logits = mx.zeros(vocab_size)
    constrained = verifier.apply(dummy_logits, state)

    for tid, name in test_tokens:
        val = constrained[tid].item()
        status = "ALLOWED" if val > -1e8 else "BLOCKED"
        print(f'  "{name}" -> {status}')

    # Demo 5: Generate constrained content
    print("\n" + "-" * 60)
    print("DEMO 5: Constrained string generation (inside JSON)")
    print("-" * 60)

    prompt = 'Complete this JSON: {"name": "'
    print(f"Prompt: {prompt}")

    result = constrained_generate(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        prompt=prompt,
        max_tokens=15,
        temperature=0.5,
        start_in_string=True,
    )
    print(f"Generated string content: {result[:50]}...")
    print("(Model generates valid string chars, verifier blocks invalid JSON)")

    print("\n" + "=" * 60)
    print("Key insight: Invalid tokens get -inf BEFORE sampling.")
    print("The model literally cannot output invalid JSON tokens.")
    print("=" * 60)


def demo_vocab_analysis():
    """
    Show what the vocab analyzer sees.
    This runs once at init, not per-token.
    """
    from verifier import VocabAnalyzer, TokenClass
    
    print("\n" + "=" * 60)
    print("VOCAB ANALYSIS DEMO")
    print("=" * 60)
    
    # Load just tokenizer
    _, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    
    # Analyze vocab
    print("\nAnalyzing vocabulary...")
    vocab = VocabAnalyzer(tokenizer)
    
    # Show class distributions
    print(f"\nVocab size: {vocab.vocab_size}")
    print("\nToken class distributions:")
    
    for cls in TokenClass:
        mask = vocab.get_class_mask(cls)
        count = int(mx.sum(mask).item())
        print(f"  {cls.name}: {count} tokens ({100*count/vocab.vocab_size:.1f}%)")
    
    # Show some example tokens per class
    print("\nExample tokens by class:")
    for cls in [TokenClass.DIGIT, TokenClass.LBRACE, TokenClass.QUOTE]:
        examples = []
        for token_id, info in vocab.token_info.items():
            if cls in info.classes and len(examples) < 5:
                examples.append(repr(info.text))
        print(f"  {cls.name}: {', '.join(examples)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--vocab":
        demo_vocab_analysis()
    else:
        demo_parameter_extraction()
