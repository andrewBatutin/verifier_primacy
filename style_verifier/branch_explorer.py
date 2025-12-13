"""CLI tool to explore branching token paths - "what if" scenarios."""

import sys

import mlx.core as mx
from mlx_lm import load
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def generate_greedy(model, tokenizer, tokens: mx.array, max_tokens: int = 50) -> str:
    """Continue greedy generation from given token sequence."""
    prompt_len = len(tokens)

    for _ in range(max_tokens):
        logits = model(tokens[None])
        next_logits = logits[0, -1, :]

        next_token = mx.argmax(next_logits)

        if next_token.item() == tokenizer.eos_token_id:
            break

        tokens = mx.concatenate([tokens, next_token[None]])

    # Return only generated part (excluding prompt)
    generated_ids = tokens[prompt_len:].tolist()
    return tokenizer.decode(generated_ids)


def branch_generate(model, tokenizer, prompt: str, num_branches: int = 3, max_tokens: int = 50):
    """Generate multiple branches from different first-token choices."""
    tokens = mx.array(tokenizer.encode(prompt))

    # Get first token logits
    logits = model(tokens[None])[0, -1, :]

    # Get top-N choices
    top_indices = mx.argsort(logits)[-num_branches:][::-1].tolist()
    top_logits = [logits[idx].item() for idx in top_indices]

    branches = []
    for idx, (token_id, logit_val) in enumerate(zip(top_indices, top_logits)):
        first_token_str = tokenizer.decode([token_id])

        # Start branch with this token
        branch_tokens = mx.concatenate([tokens, mx.array([token_id])])

        # Continue generation
        generated = generate_greedy(model, tokenizer, branch_tokens, max_tokens - 1)

        branches.append({
            "branch_num": idx + 1,
            "first_token_id": token_id,
            "first_token": first_token_str,
            "first_logit": logit_val,
            "full_text": first_token_str + generated,
        })

    return branches


def display_branches(console: Console, prompt: str, branches: list):
    """Display all branches with Rich panels."""
    console.print()
    console.print(f"[bold cyan]Prompt:[/] \"{prompt}\"")
    console.print()

    # First show the branching decision table
    table = Table(title="First Token Choices")
    table.add_column("Branch", justify="center", style="bold")
    table.add_column("Token ID", justify="right", style="cyan")
    table.add_column("Token", justify="left", style="green")
    table.add_column("Logit", justify="right", style="yellow")

    for b in branches:
        table.add_row(
            str(b["branch_num"]),
            str(b["first_token_id"]),
            repr(b["first_token"]),
            f"{b['first_logit']:.4f}"
        )

    console.print(table)
    console.print()

    # Then show each branch's full output
    for b in branches:
        title = f"Branch {b['branch_num']}: {repr(b['first_token'])} (logit: {b['first_logit']:.2f})"
        panel = Panel(
            b["full_text"],
            title=title,
            border_style="green" if b["branch_num"] == 1 else "blue" if b["branch_num"] == 2 else "magenta"
        )
        console.print(panel)
        console.print()


def main():
    """Main CLI entry point."""
    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python branch_explorer.py \"Your prompt here\" [num_branches] [max_tokens][/]")
        console.print("[dim]Example: python branch_explorer.py \"Why most startups fail:\" 3 50[/]")
        sys.exit(1)

    prompt = sys.argv[1]
    num_branches = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    console.print("[yellow]Loading model...[/]")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    console.print("[green]Model loaded![/]")

    console.print(f"[yellow]Branching into {num_branches} paths...[/]")
    branches = branch_generate(model, tokenizer, prompt, num_branches, max_tokens)

    display_branches(console, prompt, branches)


if __name__ == "__main__":
    main()
