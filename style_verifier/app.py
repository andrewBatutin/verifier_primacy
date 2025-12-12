"""
Streamlit App: Style Verifier A/B Test

Compare text generation with and without style masking side-by-side.
"""

import re
import streamlit as st
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from style_verifier import StyleVerifier
from generate import make_logits_processor


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


def generate_text(prompt: str, model, tokenizer, temp: float, max_tokens: int, use_verifier: bool, verifier: StyleVerifier) -> str:
    """Generate text with optional style verification."""
    sampler = make_sampler(temp=temp)

    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "sampler": sampler,
    }

    if use_verifier:
        processor = make_logits_processor(verifier)
        kwargs["logits_processors"] = [processor]

    return generate(**kwargs)


def highlight_banned(text: str, banned_phrases: list[str]) -> str:
    """Highlight banned phrases in text using HTML."""
    result = text
    for phrase in banned_phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub(f'<span style="background-color: #ff6b6b; color: white; padding: 0 4px; border-radius: 3px;">{phrase}</span>', result)
    return result


def main():
    st.title("🎭 Style Verifier A/B Test")
    st.markdown("*Compare generation with and without banned phrase filtering*")

    st.divider()

    # Load resources
    model, tokenizer = load_model()
    verifier = load_verifier()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")

        temp = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)

        st.divider()

        st.header("🚫 Banned Phrases")
        for phrase in verifier.banned_seqs.keys():
            st.code(phrase, language=None)

    # Main content
    col_input, _ = st.columns([3, 1])

    with col_input:
        prompt = st.text_area(
            "Enter your prompt:",
            value="The problem with AI metrics is",
            height=100,
            placeholder="Enter a prompt to test...",
        )

    generate_btn = st.button("🚀 Generate Both", type="primary", use_container_width=True)

    if generate_btn and prompt.strip():
        col_raw, col_filtered = st.columns(2)

        with col_raw:
            st.subheader("❌ Without Style Masking")
            with st.spinner("Generating raw output..."):
                raw_output = generate_text(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    temp=temp,
                    max_tokens=max_tokens,
                    use_verifier=False,
                    verifier=verifier,
                )

            # Check for banned phrases
            banned_found = []
            for phrase in verifier.banned_seqs.keys():
                if phrase.lower() in raw_output.lower():
                    banned_found.append(phrase)

            if banned_found:
                st.error(f"Found {len(banned_found)} banned phrase(s)!")
                highlighted = highlight_banned(raw_output, banned_found)
                st.markdown(highlighted, unsafe_allow_html=True)
            else:
                st.success("No banned phrases found")
                st.markdown(raw_output)

        with col_filtered:
            st.subheader("✅ With Style Masking")
            with st.spinner("Generating filtered output..."):
                filtered_output = generate_text(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    temp=temp,
                    max_tokens=max_tokens,
                    use_verifier=True,
                    verifier=verifier,
                )

            # Verify no banned phrases
            banned_found_filtered = []
            for phrase in verifier.banned_seqs.keys():
                if phrase.lower() in filtered_output.lower():
                    banned_found_filtered.append(phrase)

            if banned_found_filtered:
                st.warning(f"Unexpected: found {len(banned_found_filtered)} banned phrase(s)")
                st.markdown(filtered_output)
            else:
                st.success("Clean output - no banned phrases")
                st.markdown(filtered_output)

    elif generate_btn:
        st.warning("Please enter a prompt first.")

    # Footer
    st.divider()
    st.markdown("""
    ### How it works

    **Style Masking** prevents banned phrases at generation time by modifying logits:

    1. 📝 Same prompt goes to both generators
    2. 🔓 **Raw**: Normal generation - model can output anything
    3. 🔒 **Filtered**: Logits processor sets banned token continuations to `-inf`
    4. 🎯 Result: Filtered output literally *cannot* contain banned phrases
    """)


if __name__ == "__main__":
    main()
