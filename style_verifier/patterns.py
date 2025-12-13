"""Slop patterns and style redirects."""

PATTERNS = {
    "REVEALER_SLOP": {
        "trigger": ["CONNECTOR?", "REVEALER", "OBJECT"],
        "examples": ["And here's what", "But here's the thing", "Here's what nobody"],
        "ban_category": "OBJECT",
        "redirects": [
            "The data shows",
            "Measured:",
            "Three factors:",
            "The breakdown:",
        ],
    },

    "HYPE_UNLOCK": {
        "trigger": ["HYPE", "UNLOCK", "COPULA"],
        "examples": ["The real unlock is", "The actual secret is"],
        "ban_category": "UNLOCK",
        "redirects": [
            "The mechanism:",
            "How it works:",
            "Two components:",
            "The constraint:",
        ],
    },

    "DEATH_CLAIM": {
        "trigger": ["*", "COPULA", "DEATH"],
        "examples": ["SEO is dead", "Email is dead"],
        "ban_category": "DEATH",
        "redirects": [
            "is declining.",
            "share dropped:",
            "usage: 80% -> 20%",
        ],
    },

    "FAKE_HUMILITY": {
        "trigger": ["FAKE_HUMBLE"],
        "examples": ["I was wrong", "I used to think"],
        "ban_category": "FAKE_HUMBLE",
        "redirects": [
            "Old model:",
            "Previously:",
            "V1:",
            "Before:",
        ],
    },

    "EXPLAINER_CRINGE": {
        "trigger": ["EXPLAINER"],
        "examples": ["Let me explain", "In this article"],
        "ban_category": "EXPLAINER",
        "redirects": [
            "Summary:",
            "Key points:",
            "Overview:",
        ],
    },

    "STUDY_SLOP": {
        "trigger": ["STUDY_OPENER"],
        "examples": ["A study of 1000+", "A recent study shows", "Research shows"],
        "ban_category": "STUDY_OPENER",
        "redirects": [
            "In my humble opinion",
            "Based on my experience",
            "From what I've seen",
            "My observation:",
        ],
    },
}
