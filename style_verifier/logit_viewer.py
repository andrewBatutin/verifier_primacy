"""CLI tool to visualize top-10 logits step-by-step during generation."""

import sys

import mlx.core as mx
from mlx_lm import load
from rich.console import Console
from rich.table import Table


def get_top_k_logits(logits: mx.array, k: int = 10):
    """Extract top-k token indices and their logit values."""
    # Get indices of top k values
    top_indices = mx.argsort(logits)[-k:][::-1]
    top_values = logits[top_indices]
    return top_indices.tolist(), top_values.tolist()


def step_generate(model, tokenizer, prompt: str, max_tokens: int = 50):
    """Generate tokens one at a time, yielding logit info at each step."""
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    prompt_len = len(tokens)

    for step in range(max_tokens):
        # Forward pass - get logits for next token
        logits = model(tokens[None])
        next_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Get top-10 predictions
        top_indices, top_values = get_top_k_logits(next_logits, k=10)

        # Decode tokens for display
        top_tokens = []
        for idx, val in zip(top_indices, top_values):
            token_str = tokenizer.decode([idx])
            top_tokens.append((idx, token_str, val))

        # Get generated text so far (excluding prompt)
        generated_ids = tokens[prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids) if generated_ids else ""

        yield {
            "step": step + 1,
            "generated": generated_text,
            "top_tokens": top_tokens,
            "prompt": prompt,
        }

        # Sample next token (greedy - argmax)
        next_token = mx.argmax(next_logits)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to sequence
        tokens = mx.concatenate([tokens, next_token[None]])


def display_step(console: Console, step_data: dict, max_tokens: int):
    """Display a single generation step with logit table."""
    console.clear()

    # Header
    console.print(f"[bold cyan]Prompt:[/] {step_data['prompt']}")
    console.print()
    console.print(
        f"[bold]Step {step_data['step']}/{max_tokens}[/] | "
        f"[green]Generated:[/] \"{step_data['generated']}\""
    )
    console.print()

    # Create logits table
    table = Table(title="Top 10 Logits")
    table.add_column("Rank", justify="center", style="dim")
    table.add_column("Token ID", justify="right", style="cyan")
    table.add_column("Token", justify="left", style="green")
    table.add_column("Logit", justify="right", style="yellow")

    for rank, (token_id, token_str, logit_val) in enumerate(step_data["top_tokens"], 1):
        # Escape special characters and show repr for clarity
        display_str = repr(token_str)
        table.add_row(str(rank), str(token_id), display_str, f"{logit_val:.4f}")

    console.print(table)
    console.print()
    console.print("[dim][Enter] next | [q] quit | [j] jump to step | [r] run to end[/]")


def get_input():
    """Get single character input without requiring Enter."""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    except ImportError:
        # Fallback for Windows
        return input() or "\n"


def main():
    """Main CLI entry point."""
    console = Console()

    # Parse arguments
    if len(sys.argv) < 2:
        console.print("[red]Usage: python logit_viewer.py \"Your prompt here\"[/]")
        console.print("[dim]Example: python logit_viewer.py \"Why most startups fail:\"[/]")
        sys.exit(1)

    prompt = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Load model
    console.print("[yellow]Loading model...[/]")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    console.print("[green]Model loaded![/]")

    # Pre-generate all steps
    console.print("[yellow]Pre-generating tokens...[/]")
    steps = list(step_generate(model, tokenizer, prompt, max_tokens))
    total_steps = len(steps)
    console.print(f"[green]Generated {total_steps} steps[/]")

    # Interactive viewer
    current_step = 0

    while current_step < total_steps:
        display_step(console, steps[current_step], total_steps)

        key = get_input()

        if key in ("\r", "\n", " ", "n"):
            # Next step
            current_step += 1
        elif key == "p" and current_step > 0:
            # Previous step
            current_step -= 1
        elif key == "q":
            # Quit
            console.print("\n[yellow]Exiting...[/]")
            break
        elif key == "j":
            # Jump to step
            console.print("\n[cyan]Jump to step (1-{}): [/]".format(total_steps), end="")
            try:
                # Need to temporarily restore terminal for input
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                step_num = int(input())
                if 1 <= step_num <= total_steps:
                    current_step = step_num - 1
            except (ValueError, ImportError):
                pass
        elif key == "r":
            # Run to end - show final state
            current_step = total_steps - 1
        elif key == "0":
            # Go to start
            current_step = 0

    # Show final result
    if current_step >= total_steps:
        console.print("\n[bold green]Generation complete![/]")
        console.print(f"[cyan]Final output:[/] {steps[-1]['generated']}")


if __name__ == "__main__":
    main()
