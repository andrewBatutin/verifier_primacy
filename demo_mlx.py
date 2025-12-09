#!/usr/bin/env python3
"""
Demo: Form Filling with Constrained Fields

Scenario: User pastes text and wants LLM to extract data into a form.
Some fields have dropdown-style constraints (allowed values only).

The verifier ensures constrained fields can ONLY contain allowed values.
No hallucination, no invalid data - guaranteed at generation time.
"""

import mlx.core as mx
from mlx_lm import load

from verifier import create_allowed_values_verifier


def demo_form_filling():
    """
    Demo: IT Support Ticket Form with constrained fields.
    """
    print("=" * 70)
    print("VERIFIER PRIMACY DEMO: Form Filling with Constraints")
    print("=" * 70)

    # The input text - a realistic messy email/ticket
    input_text = """
From: sarah.jones@company.com
Subject: HELP!!! System crashed during client demo

Hi Support Team,

I'm freaking out here. Was in the middle of a super important demo with
Acme Corp (potential $2M deal!) and my whole system just froze. Blue screen
of death. I've tried rebooting 3 times but it keeps happening.

I think it might be related to that Windows update IT pushed last night?
Or maybe my RAM is failing? The laptop is making weird clicking sounds too.

This is EXTREMELY URGENT - I have a follow-up call with the client in 2 hours
and I absolutely cannot miss it. Please escalate to whoever can help fastest.

I'm in the Chicago office, desk 4B-221. Phone: x4455
Employee ID: SJ-98234
Department: Enterprise Sales

Desperately need help,
Sarah

P.S. - I also noticed Outlook has been running super slow all week,
emails taking forever to load. Not sure if related.
""".strip()

    # Form schema with constrained fields - realistic business form
    form_schema = {
        "category": {
            "type": "constrained",
            "allowed": ["hardware", "software", "network", "security", "access", "other"],
            "description": "Issue category"
        },
        "severity": {
            "type": "constrained",
            "allowed": ["low", "medium", "high", "critical"],
            "description": "Business impact level"
        },
        "affected_system": {
            "type": "constrained",
            "allowed": ["laptop", "desktop", "server", "printer", "phone", "other"],
            "description": "Primary affected system"
        },
        "escalation": {
            "type": "constrained",
            "allowed": ["none", "team_lead", "manager", "director", "vp"],
            "description": "Escalation level needed"
        },
    }

    print("\n📄 INPUT TEXT:")
    print("-" * 70)
    print(input_text)

    print("\n📋 FORM SCHEMA:")
    print("-" * 70)
    for field, config in form_schema.items():
        if config["type"] == "constrained":
            print(f"  {field}: {config['allowed']}")

    # Load model (small 0.6B model - verifier still works!)
    print("\n⏳ Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit")
    vocab_size = model.model.embed_tokens.weight.shape[0]

    print("\n" + "=" * 70)
    print("FIELD-BY-FIELD EXTRACTION WITH VERIFICATION")
    print("=" * 70)

    # Process each constrained field
    for field_name, config in form_schema.items():
        allowed_values = config["allowed"]

        print(f"\n{'─' * 70}")
        print(f"📝 FIELD: {field_name}")
        print(f"   Allowed values: {allowed_values}")
        print(f"{'─' * 70}")

        # Create verifier for this field
        # Include space-prefixed versions since tokenizers often have " word" tokens
        expanded_values = allowed_values + [f" {v}" for v in allowed_values]
        verifier = create_allowed_values_verifier(
            tokenizer=tokenizer,
            allowed_values=expanded_values,
            vocab_size=vocab_size,
        )

        # Construct prompt for extraction
        prompt = f"""Extract the {field_name} from this text.
Text: {input_text}

The {field_name} must be one of: {', '.join(allowed_values)}
{field_name}:"""

        prompt_tokens = mx.array(tokenizer.encode(prompt))

        # Get model's logits
        logits = model(prompt_tokens[None, :])[:, -1, :].squeeze(0)

        # 1. Show UNCONSTRAINED top predictions
        print(f"\n   🔓 WITHOUT constraint (model's raw preferences):")
        top_unconstrained = mx.argsort(logits)[-8:][::-1].tolist()
        for i, tid in enumerate(top_unconstrained[:5]):
            txt = tokenizer.decode([tid]).strip()
            print(f"      #{i+1}: '{txt}' (logit={logits[tid].item():.1f})")

        # 2. Apply verifier
        constrained_logits = verifier.apply(logits, verifier.create_initial_state())

        # 3. Show CONSTRAINED top predictions
        print(f"\n   🔒 WITH constraint (only allowed values):")
        top_constrained = mx.argsort(constrained_logits)[-5:][::-1].tolist()
        for i, tid in enumerate(top_constrained):
            txt = tokenizer.decode([tid]).strip()
            logit_val = constrained_logits[tid].item()
            if logit_val > -1e8:
                print(f"      #{i+1}: '{txt}' (logit={logit_val:.1f}) ✓")

        # 4. Show what gets BLOCKED - real tokens from model's top choices
        print(f"\n   🚫 BLOCKED (high-probability tokens that are not allowed):")
        # Get top 30 from unconstrained to find blocked ones
        top30 = mx.argsort(logits)[-30:][::-1].tolist()
        blocked_count = 0
        for tid in top30:
            if blocked_count >= 5:
                break
            masked = constrained_logits[tid].item()
            if masked < -1e8:  # This token was blocked
                txt = tokenizer.decode([tid]).replace('\n', '\\n')
                # Skip pure whitespace/newline tokens for cleaner output
                if txt.strip() == '' or txt == '\\n' or txt == '\\n\\n':
                    continue
                orig = logits[tid].item()
                print(f"      '{txt.strip()}' (logit={orig:.1f}) → -inf BLOCKED")
                blocked_count += 1

        # 5. Final selection
        selected_token = mx.argmax(constrained_logits).item()
        selected_text = tokenizer.decode([selected_token]).strip()
        print(f"\n   ✅ SELECTED: {selected_text}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The model extracts information from text, but constrained fields can
ONLY contain values from the allowed list.

For 'priority': Even if model wants to output "urgent" (from the text),
it's blocked. Only "low", "medium", "high", "critical" are possible.

This is VERIFICATION AT GENERATION TIME:
- Not: Generate → Validate → Retry if invalid
- But: Constrain → Generate valid output guaranteed

The invalid tokens get -inf logits, making them impossible to sample.
""")


if __name__ == "__main__":
    demo_form_filling()
