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
                  mode: str, verifier: StyleVerifier, boost: float = 5.0) -> str:
    """Generate text with specified style control mode."""
    sampler = make_sampler(temp=temp)

    processors = []

    if mode in ["block", "block+redirect"]:
        processors.append(make_logits_processor(verifier))

    if mode == "block+redirect":
        processors.append(make_redirect_processor(boost=boost))

    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "sampler": sampler,
    }

    if processors:
        kwargs["logits_processors"] = processors

    return generate(**kwargs)


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
        st.header("⚙️ Settings")

        temp = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)
        boost = st.slider("Redirect Boost", 1.0, 20.0, 5.0, 1.0)

        st.divider()

        st.header("🚫 Banned Phrases")
        for phrase in list(verifier.banned_seqs.keys())[:10]:  # First 10
            st.code(phrase, language=None)
        if len(verifier.banned_seqs) > 10:
            st.caption(f"... and {len(verifier.banned_seqs) - 10} more")

    # Main content
    prompt = st.text_area(
        "Enter your prompt:",
        value="Why most startups fail:",
        height=100,
        placeholder="Enter a prompt to test...",
    )

    generate_btn = st.button("🚀 Generate All Modes", type="primary", use_container_width=True)

    if generate_btn and prompt.strip():
        col_raw, col_block, col_redirect = st.columns(3)

        modes = [
            ("raw", col_raw, "❌ Raw", "No filtering"),
            ("block", col_block, "🚫 Block Only", "Exact phrase masking"),
            ("block+redirect", col_redirect, "✨ Block + Redirect", "Masking + style boost"),
        ]

        for mode, col, title, desc in modes:
            with col:
                st.subheader(title)
                st.caption(desc)

                with st.spinner("Generating..."):
                    output = generate_text(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        temp=temp,
                        max_tokens=max_tokens,
                        mode=mode,
                        verifier=verifier,
                        boost=boost,
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

    elif generate_btn:
        st.warning("Please enter a prompt first.")

    # Footer
    st.divider()
    st.markdown("""
    ### Modes

    | Mode | Description |
    |------|-------------|
    | **Raw** | No filtering - baseline output |
    | **Block Only** | Exact phrase masking (logits → -inf) |
    | **Block + Redirect** | Masking + boost style alternatives |
    """)


if __name__ == "__main__":
    main()
