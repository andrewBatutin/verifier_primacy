"""
Streamlit App: Style Verifier A/B Test

Compare text generation with different style control modes.
"""

import re
import streamlit as st
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from style_verifier import StyleVerifier
from generate import make_logits_processor
from style_redirect import make_redirect_processor
from advanced_masking import make_advanced_processor


st.set_page_config(
    page_title="Style Verifier A/B Test",
    page_icon="🎭",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load model once and cache it."""
    with st.spinner("Loading AI model (Qwen3-4B)..."):
        model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    return model, tokenizer


@st.cache_resource
def load_verifier():
    """Load style verifier and cache it."""
    return StyleVerifier()


def generate_text(prompt: str, model, tokenizer, temp: float, max_tokens: int,
                  mode: str, verifier: StyleVerifier, boost: float = 5.0,
                  advanced_config: dict = None) -> tuple[str, dict]:
    """Generate text with specified style control mode. Returns (text, metrics)."""
    sampler = make_sampler(temp=temp)

    processors = []
    metrics = {}

    if mode == "raw":
        pass  # No processors
    elif mode == "block":
        processors.append(make_logits_processor(verifier))
    elif mode == "block+redirect":
        processors.append(make_logits_processor(verifier))
        processors.append(make_redirect_processor(boost=boost))
    elif mode == "advanced" and advanced_config:
        proc = make_advanced_processor(model, tokenizer, advanced_config, verifier)
        processors.append(proc)
        # Store reference to get metrics after generation
        metrics["processor"] = proc

    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "sampler": sampler,
    }

    if processors:
        kwargs["logits_processors"] = processors

    output = generate(**kwargs)

    # Get metrics from advanced processor
    if "processor" in metrics and hasattr(metrics["processor"], "processor"):
        metrics["last"] = metrics["processor"].processor.last_metrics

    return output, metrics


def highlight_banned(text: str, banned_phrases: list[str]) -> str:
    """Highlight banned phrases in text using HTML."""
    result = text
    for phrase in banned_phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub(
            f'<span style="background-color: #ff6b6b; color: white; padding: 0 4px; border-radius: 3px;">{phrase}</span>',
            result
        )
    return result


def main():
    st.title("🎭 Style Verifier A/B Test")
    st.markdown("*Compare generation with different style control modes*")

    st.divider()

    # Load resources
    model, tokenizer = load_model()
    verifier = load_verifier()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Basic Settings")

        temp = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)
        boost = st.slider("Redirect Boost", 1.0, 20.0, 5.0, 1.0)

        st.divider()

        # Advanced Masking Controls
        st.header("🔬 Advanced Masking")
        st.caption("Toggle techniques for the Advanced column")

        use_hard_mask = st.checkbox("Hard Masking (-inf)", value=True,
                                    help="Set banned tokens to -infinity (guaranteed block)")
        use_soft_mask = st.checkbox("Soft Masking (penalty)", value=False,
                                    help="Apply graduated penalty instead of -inf")
        use_embedding = st.checkbox("Embedding Detection", value=False,
                                    help="Detect slop by semantic similarity")
        use_entropy = st.checkbox("Entropy-Adaptive", value=False,
                                  help="Only intervene when model is uncertain")
        use_lookahead = st.checkbox("Lookahead Detection", value=False,
                                    help="Simulate future tokens to detect slop (slow)")
        use_boost_alt = st.checkbox("Boost Alternatives", value=False,
                                    help="Boost tokens that start clean phrases")

        # Parameters (collapsible)
        with st.expander("⚡ Advanced Parameters"):
            soft_penalty = st.slider("Soft Penalty", -20.0, -1.0, -10.0, 1.0)
            embedding_threshold = st.slider("Embedding Threshold", 0.5, 0.95, 0.80, 0.05)
            entropy_threshold = st.slider("Entropy Threshold", 0.5, 4.0, 2.0, 0.5)
            lookahead_depth = st.slider("Lookahead Depth", 1, 5, 3, 1)
            boost_strength = st.slider("Boost Strength", 1.0, 20.0, 5.0, 1.0)

        st.divider()

        st.header("🚫 Banned Phrases")
        for phrase in list(verifier.banned_seqs.keys())[:8]:
            st.code(phrase, language=None)
        if len(verifier.banned_seqs) > 8:
            st.caption(f"... and {len(verifier.banned_seqs) - 8} more")

    # Build advanced config
    advanced_config = {
        "hard_mask": use_hard_mask and not use_soft_mask,
        "soft_mask": use_soft_mask,
        "embedding": use_embedding,
        "entropy_adaptive": use_entropy,
        "lookahead": use_lookahead,
        "boost_alternatives": use_boost_alt,
        "soft_penalty": soft_penalty,
        "embedding_threshold": embedding_threshold,
        "entropy_threshold": entropy_threshold,
        "lookahead_depth": lookahead_depth,
        "boost_strength": boost_strength,
    }

    # Main content
    prompt = st.text_area(
        "Enter your prompt:",
        value="Why most startups fail:",
        height=100,
        placeholder="Enter a prompt to test...",
    )

    generate_btn = st.button("🚀 Generate All Modes", type="primary", use_container_width=True)

    if generate_btn and prompt.strip():
        # Show active advanced techniques
        active_techniques = [k for k, v in advanced_config.items()
                            if v is True and k not in ["hard_mask"]]
        if active_techniques:
            st.info(f"**Advanced techniques active:** {', '.join(active_techniques)}")

        col_raw, col_block, col_redirect, col_advanced = st.columns(4)

        modes = [
            ("raw", col_raw, "❌ Raw", "No filtering"),
            ("block", col_block, "🚫 Block", "Hard mask"),
            ("block+redirect", col_redirect, "✨ Redirect", "Block + boost"),
            ("advanced", col_advanced, "🔬 Advanced", "Custom mix"),
        ]

        for mode, col, title, desc in modes:
            with col:
                st.subheader(title)
                st.caption(desc)

                with st.spinner("Generating..."):
                    output, metrics = generate_text(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        temp=temp,
                        max_tokens=max_tokens,
                        mode=mode,
                        verifier=verifier,
                        boost=boost,
                        advanced_config=advanced_config if mode == "advanced" else None,
                    )

                # Check for banned phrases
                banned_found = []
                for phrase in verifier.banned_seqs.keys():
                    if phrase.lower() in output.lower():
                        banned_found.append(phrase)

                if banned_found:
                    st.error(f"Found {len(banned_found)} banned")
                    highlighted = highlight_banned(output, banned_found)
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.success("Clean")
                    st.markdown(output)

                # Show metrics for advanced mode
                if mode == "advanced" and "last" in metrics:
                    m = metrics["last"]
                    with st.expander("📊 Metrics"):
                        st.write(f"**Entropy:** {m.get('entropy', 0):.2f}")
                        st.write(f"**Embedding sim:** {m.get('embedding_sim', 0):.2f}")
                        if m.get("detected_pattern"):
                            st.write(f"**Pattern:** {m['detected_pattern']}")
                        st.write(f"**Applied:** {', '.join(m.get('techniques_applied', []))}")

    elif generate_btn:
        st.warning("Please enter a prompt first.")

    # Footer
    st.divider()
    st.markdown("""
    ### Modes

    | Mode | Description |
    |------|-------------|
    | **Raw** | No filtering - baseline output |
    | **Block** | Exact phrase masking (logits → -inf) |
    | **Redirect** | Masking + boost style alternatives |
    | **Advanced** | Configurable mix of techniques |

    ### Advanced Techniques

    | Technique | Description |
    |-----------|-------------|
    | **Hard Mask** | Set banned tokens to -infinity (guaranteed) |
    | **Soft Mask** | Graduated penalty (allows rare escape) |
    | **Embedding** | Detect slop by semantic similarity |
    | **Entropy-Adaptive** | Only intervene when uncertain |
    | **Lookahead** | Simulate future tokens (slow) |
    | **Boost Alternatives** | Increase probability of clean phrases |
    """)


if __name__ == "__main__":
    main()
