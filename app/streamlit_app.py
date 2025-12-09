"""
Streamlit App: Insurance Claim Processor with Verifier Primacy

A web UI demonstrating constrained form filling from natural language.
"""

import streamlit as st
import mlx.core as mx
from mlx_lm import load

from verifier import create_allowed_values_verifier


# Page config
st.set_page_config(
    page_title="Insurance Claim Processor",
    page_icon="🏥",
    layout="wide",
)


# Form schema
FORM_SCHEMA = {
    "claim_type": {
        "allowed": ["auto", "home", "health", "life", "travel"],
        "icon": "🚗",
        "description": "Type of insurance claim",
    },
    "incident": {
        "allowed": ["collision", "theft", "vandalism", "weather", "fire", "medical"],
        "icon": "⚠️",
        "description": "What happened",
    },
    "fault": {
        "allowed": ["claimant", "other_party", "shared", "undetermined"],
        "icon": "⚖️",
        "description": "Who is responsible",
    },
    "injury": {
        "allowed": ["none", "minor", "moderate", "severe"],
        "icon": "🩹",
        "description": "Injury severity",
    },
    "police_report": {
        "allowed": ["yes", "no"],
        "icon": "👮",
        "description": "Was police involved",
    },
}


@st.cache_resource
def load_model():
    """Load model once and cache it."""
    with st.spinner("Loading AI model (Qwen3-4B)..."):
        model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
        vocab_size = model.model.embed_tokens.weight.shape[0]
    return model, tokenizer, vocab_size


@st.cache_resource
def create_verifiers(_tokenizer, vocab_size):
    """Create verifiers for each field."""
    verifiers = {}
    for field, config in FORM_SCHEMA.items():
        expanded = config["allowed"] + [f" {v}" for v in config["allowed"]]
        verifiers[field] = create_allowed_values_verifier(
            tokenizer=_tokenizer,
            allowed_values=expanded,
            vocab_size=vocab_size,
        )
    return verifiers


def extract_field(model, tokenizer, verifier, field_name, allowed_values, user_text):
    """Extract a single field value with verification."""
    prompt = f"""Extract the {field_name} from this insurance claim.

Claim: {user_text}

The {field_name} must be exactly one of: {', '.join(allowed_values)}
Based on the claim, the {field_name} is:"""

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    logits = model(prompt_tokens[None, :])[:, -1, :].squeeze(0)
    constrained = verifier.apply(logits, verifier.create_initial_state())

    # Get blocked tokens
    top20 = mx.argsort(logits)[-20:][::-1].tolist()
    blocked = []
    for tid in top20:
        if constrained[tid].item() < -1e8:
            txt = tokenizer.decode([tid]).strip()
            if txt and len(txt) > 1 and txt not in ['\n', '\n\n', '?', '??', '???']:
                blocked.append(txt)
                if len(blocked) >= 3:
                    break

    # Select best valid token
    selected_id = mx.argmax(constrained).item()
    selected = tokenizer.decode([selected_id]).strip()

    return {"value": selected, "blocked": blocked}


def main():
    # Header
    st.title("🏥 Insurance Claim Processor")
    st.markdown("*Powered by Verifier Primacy - Constrained AI Form Filling*")

    st.divider()

    # Load model
    model, tokenizer, vocab_size = load_model()
    verifiers = create_verifiers(tokenizer, vocab_size)

    # Two columns layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Describe Your Claim")

        # Example claims
        examples = {
            "Select an example...": "",
            "🚗 Auto Collision": "I was rear-ended at a red light yesterday on Main Street. The other driver hit me from behind while I was stopped. My bumper is completely smashed and I have whiplash - my neck really hurts. The police came and filed a report.",
            "🏠 Storm Damage": "A huge tree fell on my roof during last night's storm. There's a massive hole and water is leaking into my living room. No one was hurt thankfully. I haven't called the police since it was just weather damage.",
            "🚗 Car Theft": "Someone broke into my car last night and stole my laptop and wallet. Window was smashed. I filed a police report this morning. I wasn't there when it happened so I'm fine.",
            "🔥 House Fire": "There was an electrical fire in my kitchen last night. The fire department came and put it out but there's significant damage. I inhaled some smoke and went to the ER. Police and fire department filed reports.",
        }

        selected_example = st.selectbox("Quick examples:", list(examples.keys()))

        default_text = examples[selected_example] if selected_example != "Select an example..." else ""

        claim_text = st.text_area(
            "Enter your claim description:",
            value=default_text,
            height=200,
            placeholder="Describe what happened in your own words...",
        )

        process_btn = st.button("🔍 Process Claim", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Extracted Form Data")

        # Show schema
        with st.expander("ℹ️ Form Fields & Allowed Values", expanded=False):
            for field, config in FORM_SCHEMA.items():
                st.markdown(f"**{config['icon']} {field}**: `{' | '.join(config['allowed'])}`")

        if process_btn and claim_text.strip():
            results = {}

            # Progress bar
            progress = st.progress(0)
            status = st.empty()

            for i, (field, config) in enumerate(FORM_SCHEMA.items()):
                status.text(f"Extracting {field}...")
                results[field] = extract_field(
                    model=model,
                    tokenizer=tokenizer,
                    verifier=verifiers[field],
                    field_name=field,
                    allowed_values=config["allowed"],
                    user_text=claim_text,
                )
                progress.progress((i + 1) / len(FORM_SCHEMA))

            status.empty()
            progress.empty()

            # Display results
            st.success("✅ Claim processed successfully!")

            for field, config in FORM_SCHEMA.items():
                data = results[field]
                value = data["value"]
                blocked = data["blocked"]

                # Create a nice card-like display
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 3])

                    with c1:
                        st.markdown(f"**{config['icon']} {field}**")

                    with c2:
                        st.code(value, language=None)

                    with c3:
                        if blocked:
                            blocked_str = ", ".join([f'"{b}"' for b in blocked])
                            st.caption(f"🚫 blocked: {blocked_str}")
                        else:
                            st.caption("✓ no conflicts")

                st.divider()

        elif process_btn:
            st.warning("Please enter a claim description first.")

        else:
            # Show placeholder
            st.info("👆 Enter a claim description and click 'Process Claim' to see results.")

    # Footer
    st.divider()
    st.markdown("""
    ### How it works

    **Verifier Primacy** constrains the AI model's output at generation time:

    1. 📝 You describe your claim in natural language
    2. 🤖 The AI model generates logits (probability distribution over all tokens)
    3. 🔒 The **Verifier** masks invalid tokens to `-infinity`
    4. ✅ Only valid form values can be selected

    This is **not** post-generation validation - invalid values are **impossible** to generate.
    """)


if __name__ == "__main__":
    main()
