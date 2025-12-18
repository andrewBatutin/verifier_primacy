# Verifier Primacy - Project Restructure for Claude Code

## Current State
```
verifier_primacy/
├── app/
├── notebooks/
├── style_verifier/
├── cli_agent.py
├── demo_mlx.py
├── verifier.py
├── pyproject.toml
└── uv.lock
```

## Target State (Claude Code Optimized)
```
verifier_primacy/
├── .claude/
│   ├── CLAUDE.md              # Main instructions (imports others)
│   ├── settings.json          # Hooks config
│   ├── commands/              # Custom slash commands
│   │   ├── test.md           # /project:test
│   │   ├── bench.md          # /project:bench
│   │   ├── demo.md           # /project:demo
│   │   └── release.md        # /project:release
│   ├── agents/                # Subagents
│   │   ├── reviewer.md       # Code review agent
│   │   ├── benchmarker.md    # Performance testing agent
│   │   └── docs-writer.md    # Documentation agent
│   └── rules/                 # Conditional rules
│       ├── mlx-backend.md    # Rules for MLX files
│       ├── vllm-backend.md   # Rules for vLLM files
│       └── tests.md          # Rules for test files
├── CLAUDE.md                  # Root (redirects to .claude/)
├── src/
│   └── verifier_primacy/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── confidence.py     # Entropy, calibration, top-k gap
│       │   ├── primitives.py     # Verification checks
│       │   ├── routing.py        # Human/pass/reject decisions
│       │   └── vocab.py          # Vocabulary analysis
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract backend
│       │   ├── mlx.py            # Apple Silicon
│       │   └── vllm.py           # Production servers
│       ├── rules/
│       │   ├── __init__.py
│       │   ├── base.py           # VerificationRule ABC
│       │   ├── json_structure.py
│       │   ├── schema.py
│       │   └── custom.py
│       └── schemas/
│           ├── __init__.py
│           └── pydantic.py       # Pydantic integration
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_confidence.py
│   ├── test_primitives.py
│   ├── test_routing.py
│   └── evals/                    # Like Instructor - weekly evals
│       ├── __init__.py
│       └── test_extraction.py
├── benchmarks/
│   ├── bench_overhead.py
│   └── bench_accuracy.py
├── examples/
│   ├── basic_extraction.py
│   ├── tool_calling.py
│   └── human_routing.py
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── backends.md
├── scripts/
│   ├── hooks/
│   │   ├── pre_commit.py        # Lint + type check
│   │   └── post_test.py         # Coverage report
│   └── release.py
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

## Why This Structure?

### 1. Claude Code Integration
- **CLAUDE.md hierarchy**: Root file imports detailed rules
- **Subagents**: Dedicated agents for review, benchmarking, docs
- **Custom commands**: `/project:test`, `/project:bench` for common workflows
- **Hooks**: Auto-lint, auto-test on file changes

### 2. Library Best Practices (Inspired by Instructor)
- **src layout**: Prevents import confusion
- **Separate backends**: Easy to add new inference engines
- **Evals directory**: Track quality over time
- **Examples**: Runnable demos for quick onboarding

### 3. Development Velocity
- **Hooks enforce quality**: No manual linting
- **Subagents parallelize work**: Review while you code
- **Commands standardize workflows**: Team consistency
