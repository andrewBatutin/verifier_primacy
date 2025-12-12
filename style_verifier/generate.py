"""Generate text with style verification."""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from style_verifier import StyleVerifier


def make_logits_processor(verifier: StyleVerifier):
    """Create a logits processor that applies style verification.

    Args:
        verifier: StyleVerifier instance

    Returns:
        A function compatible with mlx_lm's logits_processors parameter
    """

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply style verification to logits.

        Args:
            tokens: Generated tokens so far (shape: [seq_len])
            logits: Logits for next token (shape: [vocab_size])

        Returns:
            Modified logits with banned continuations masked
        """
        generated = tokens.tolist()
        return verifier.mask(logits, generated)

    return processor


def generate_clean(
    prompt: str,
    model,
    tokenizer,
    verifier: StyleVerifier,
    max_tokens: int = 256,
    temp: float = 0.7,
) -> str:
    """Generate text while blocking banned phrases.

    Args:
        prompt: User prompt
        model: MLX language model
        tokenizer: Tokenizer for the model
        verifier: StyleVerifier instance
        max_tokens: Maximum tokens to generate
        temp: Sampling temperature (0 = greedy)

    Returns:
        Generated text
    """
    processor = make_logits_processor(verifier)
    sampler = make_sampler(temp=temp)

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=[processor],
    )

    return result


def main():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")

    print("Loading style verifier...")
    verifier = StyleVerifier()

    prompt = "The problem with AI metrics is"
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    output = generate_clean(prompt, model, tokenizer, verifier)
    print(output)


if __name__ == "__main__":
    main()
