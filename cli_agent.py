#!/usr/bin/env python3
"""
Interactive CLI Agent: Text → Constrained Form

Type any text request and watch it get mapped to a structured form
with constrained fields. Demonstrates verifier primacy in action.
"""

import mlx.core as mx
from mlx_lm import load

from verifier import create_allowed_values_verifier


# ANSI colors for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header():
    """Print the app header."""
    print()
    print(f"{Colors.CYAN}╔══════════════════════════════════════════════════════════════════╗{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}  {Colors.BOLD}🏥 INSURANCE CLAIM PROCESSOR{Colors.END} {Colors.DIM}(with Verifier Primacy){Colors.END}          {Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}╚══════════════════════════════════════════════════════════════════╝{Colors.END}")


def print_schema(form_schema: dict):
    """Print the form schema."""
    print(f"\n{Colors.BOLD}Form Fields:{Colors.END}")
    for field, config in form_schema.items():
        values = " | ".join(config["allowed"])
        print(f"  {Colors.DIM}•{Colors.END} {field}:  {Colors.YELLOW}[{values}]{Colors.END}")
    print()


def extract_field(
    model,
    tokenizer,
    verifier,
    field_name: str,
    allowed_values: list[str],
    user_text: str,
    vocab_size: int,
) -> dict:
    """Extract a single field value with verification."""

    # Build prompt
    prompt = f"""Extract the {field_name} from this support request.

Request: {user_text}

The {field_name} must be exactly one of: {', '.join(allowed_values)}
Based on the request, the {field_name} is:"""

    prompt_tokens = mx.array(tokenizer.encode(prompt))

    # Get model logits
    logits = model(prompt_tokens[None, :])[:, -1, :].squeeze(0)

    # Apply verifier
    constrained = verifier.apply(logits, verifier.create_initial_state())

    # Get top blocked tokens (real ones model wanted)
    top20 = mx.argsort(logits)[-20:][::-1].tolist()
    blocked = []
    for tid in top20:
        if constrained[tid].item() < -1e8:
            txt = tokenizer.decode([tid]).strip()
            if txt and len(txt) > 1 and txt not in ['\n', '\n\n', '?']:
                blocked.append((txt, logits[tid].item()))
                if len(blocked) >= 3:
                    break

    # Select best valid token
    selected_id = mx.argmax(constrained).item()
    selected = tokenizer.decode([selected_id]).strip()

    return {
        "value": selected,
        "blocked": blocked,
    }


def print_results(results: dict):
    """Print extraction results in a nice box."""
    print(f"\n{Colors.GREEN}┌─────────────────────────────────────────────────────────────────┐{Colors.END}")
    print(f"{Colors.GREEN}│{Colors.END} {Colors.BOLD}EXTRACTED FORM DATA{Colors.END}                                             {Colors.GREEN}│{Colors.END}")
    print(f"{Colors.GREEN}├─────────────────────────────────────────────────────────────────┤{Colors.END}")

    for field, data in results.items():
        value = data["value"]
        blocked = data["blocked"]

        # Format blocked tokens
        if blocked:
            blocked_str = ", ".join([f'"{b[0]}"' for b in blocked[:3]])
            blocked_display = f'{Colors.DIM}(blocked: {blocked_str}){Colors.END}'
        else:
            blocked_display = ""

        # Pad field name
        field_padded = f"{field}:".ljust(14)
        value_padded = value.ljust(12)

        print(f"{Colors.GREEN}│{Colors.END}  {field_padded}{Colors.CYAN}{value_padded}{Colors.END} ✓ {blocked_display}")

    print(f"{Colors.GREEN}└─────────────────────────────────────────────────────────────────┘{Colors.END}")


def main():
    # Insurance Claim Form Schema
    form_schema = {
        "claim_type": {
            "allowed": ["auto", "home", "health", "life", "travel"],
        },
        "incident": {
            "allowed": ["collision", "theft", "vandalism", "weather", "fire", "medical"],
        },
        "fault": {
            "allowed": ["claimant", "other_party", "shared", "undetermined"],
        },
        "injury": {
            "allowed": ["none", "minor", "moderate", "severe"],
        },
        "police_report": {
            "allowed": ["yes", "no"],
        },
    }

    print_header()
    print_schema(form_schema)

    # Load model (4B for better understanding)
    print(f"{Colors.DIM}Loading model (Qwen3-4B)...{Colors.END}")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    vocab_size = model.model.embed_tokens.weight.shape[0]

    # Pre-create verifiers for each field
    verifiers = {}
    for field, config in form_schema.items():
        expanded = config["allowed"] + [f" {v}" for v in config["allowed"]]
        verifiers[field] = create_allowed_values_verifier(
            tokenizer=tokenizer,
            allowed_values=expanded,
            vocab_size=vocab_size,
        )

    print(f"{Colors.GREEN}Ready!{Colors.END}")
    print(f"\n{Colors.DIM}{'━' * 68}{Colors.END}")
    print(f"Describe your insurance claim (or {Colors.YELLOW}'quit'{Colors.END} to exit):")

    while True:
        try:
            print()
            user_input = input(f"{Colors.BOLD}> {Colors.END}").strip()

            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.DIM}Goodbye!{Colors.END}\n")
                break

            print(f"\n{Colors.DIM}Processing...{Colors.END}")

            # Extract each field
            results = {}
            for field, config in form_schema.items():
                results[field] = extract_field(
                    model=model,
                    tokenizer=tokenizer,
                    verifier=verifiers[field],
                    field_name=field,
                    allowed_values=config["allowed"],
                    user_text=user_input,
                    vocab_size=vocab_size,
                )

            print_results(results)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.DIM}Interrupted. Goodbye!{Colors.END}\n")
            break
        except EOFError:
            # End of input (piped input finished)
            print(f"\n{Colors.DIM}Done.{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
            break


if __name__ == "__main__":
    main()
