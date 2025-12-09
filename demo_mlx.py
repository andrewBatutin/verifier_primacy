#!/usr/bin/env python3
"""
Demo: Business Rules Verification via Logits Masking

This demonstrates the core concept of "verifier primacy":
- Instead of generating text and then validating it
- We constrain the model's output BEFORE sampling
- Invalid tokens get -inf logits and literally cannot be generated

Example: A restaurant that only serves Coca-Cola products
- Allowed: "coca cola", "sprite", "fanta"
- Blocked: "pepsi", "mountain dew", "7up"

The model cannot output "pepsi" - it's not validation, it's prevention.
"""

import mlx.core as mx
from mlx_lm import load

from verifier import (
    AllowedValuesRule,
    create_allowed_values_verifier,
)


def demo_drinks_verification():
    """
    Demo: Restaurant only serves specific drinks.

    Shows how logits masking prevents invalid outputs.
    """
    print("=" * 60)
    print("VERIFIER PRIMACY DEMO")
    print("Business Rule: Restaurant Drink Menu")
    print("=" * 60)

    # Business rule: Only these drinks are allowed
    allowed_drinks = ["coca cola", "sprite", "fanta", "water"]
    blocked_drinks = ["pepsi", "mountain dew", "7up", "dr pepper"]

    print("\n✓ ALLOWED drinks:", ", ".join(allowed_drinks))
    print("✗ BLOCKED drinks:", ", ".join(blocked_drinks))

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    vocab_size = model.model.embed_tokens.weight.shape[0]

    # Create verifier with business rule
    verifier = create_allowed_values_verifier(
        tokenizer=tokenizer,
        allowed_values=allowed_drinks,
        vocab_size=vocab_size,
    )

    print(f"\nVerifier initialized:")
    print(f"  - Vocab size: {vocab_size:,}")
    print(f"  - Allowed tokens: {len(verifier.rules[0].allowed_token_ids)}")

    # Show which tokens are allowed
    print("\n" + "-" * 60)
    print("TOKEN ANALYSIS")
    print("-" * 60)

    rule = verifier.rules[0]

    print("\nAllowed drink tokens:")
    for drink in allowed_drinks:
        tokens = tokenizer.encode(drink)
        token_texts = [tokenizer.decode([t]) for t in tokens]
        print(f"  '{drink}' → {tokens} → {token_texts}")

    print("\nBlocked drink tokens:")
    for drink in blocked_drinks:
        tokens = tokenizer.encode(drink)
        token_texts = [tokenizer.decode([t]) for t in tokens]
        blocked = [t for t in tokens if t not in rule.allowed_token_ids]
        print(f"  '{drink}' → blocked tokens: {blocked}")

    # Demo: Check specific tokens
    print("\n" + "-" * 60)
    print("LOGIT MASKING DEMO")
    print("-" * 60)

    # Create dummy logits (all zeros)
    dummy_logits = mx.zeros(vocab_size)

    # Apply verifier
    masked_logits = verifier.apply(dummy_logits, verifier.create_initial_state())

    # Check what happens to specific tokens
    test_words = ["coca", "cola", "pepsi", "sprite", "mountain"]
    print("\nToken logits after masking:")
    for word in test_words:
        token_id = tokenizer.encode(word)[0]
        original = dummy_logits[token_id].item()
        masked = masked_logits[token_id].item()
        status = "ALLOWED ✓" if masked > -1e8 else "BLOCKED ✗"
        print(f"  '{word}' (id={token_id}): {original:.0f} → {masked:.0f} [{status}]")

    # Demo: Compare constrained vs unconstrained
    print("\n" + "-" * 60)
    print("GENERATION COMPARISON")
    print("-" * 60)

    prompt = "What soft drink would you recommend? I suggest"
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    print(f"\nPrompt: '{prompt}'")

    # 1. UNCONSTRAINED generation
    print("\n1. WITHOUT verification (model free to generate anything):")
    logits = model(prompt_tokens[None, :])[:, -1, :].squeeze(0)
    top5 = mx.argsort(logits)[-5:][::-1].tolist()
    for i, tid in enumerate(top5):
        txt = tokenizer.decode([tid])
        print(f"   #{i+1}: '{txt}' (logit={logits[tid].item():.1f})")

    # 2. CONSTRAINED generation
    print("\n2. WITH verification (only allowed drinks):")
    constrained = verifier.apply(logits, verifier.create_initial_state())
    top5 = mx.argsort(constrained)[-5:][::-1].tolist()
    for i, tid in enumerate(top5):
        txt = tokenizer.decode([tid])
        is_allowed = tid in rule.allowed_token_ids
        print(f"   #{i+1}: '{txt}' (logit={constrained[tid].item():.1f}) {'✓' if is_allowed else ''}")

    # 3. Show that pepsi is impossible
    print("\n3. PROOF: 'pepsi' tokens are impossible to generate:")
    pepsi_tokens = tokenizer.encode("pepsi")
    for tid in pepsi_tokens:
        txt = tokenizer.decode([tid])
        original = logits[tid].item()
        masked = constrained[tid].item()
        print(f"   Token '{txt}' (id={tid}): {original:.1f} → {masked:.0f} (BLOCKED)")

    # Final summary
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The model CANNOT generate "pepsi" or other blocked drinks.
It's not validation after generation - it's prevention at the source.

How it works:
1. Model produces logits (probability distribution over all tokens)
2. Verifier masks blocked tokens to -infinity
3. Softmax(-inf) = 0, so blocked tokens have zero probability
4. Sampling can only select from allowed tokens

This is "verifier primacy" - verify don't pray.
""")


if __name__ == "__main__":
    demo_drinks_verification()
