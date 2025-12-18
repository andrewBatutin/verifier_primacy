#!/usr/bin/env python3
"""Migration script to restructure verifier_primacy for Claude Code.

Run this script from the root of your existing verifier_primacy repo:
    python migrate_structure.py

It will:
1. Create the new directory structure
2. Move existing files to new locations
3. Create placeholder files where needed
4. Set up Claude Code configuration

Backup your repo first!
"""

import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create the new directory structure."""
    dirs = [
        ".claude/commands",
        ".claude/agents",
        ".claude/rules",
        "src/verifier_primacy/core",
        "src/verifier_primacy/backends",
        "src/verifier_primacy/rules",
        "src/verifier_primacy/schemas",
        "tests/evals",
        "benchmarks",
        "examples",
        "docs",
        "scripts/hooks",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("Created: %s/", d)


def create_init_files():
    """Create __init__.py files."""
    init_locations = [
        "src/verifier_primacy/__init__.py",
        "src/verifier_primacy/core/__init__.py",
        "src/verifier_primacy/backends/__init__.py",
        "src/verifier_primacy/rules/__init__.py",
        "src/verifier_primacy/schemas/__init__.py",
        "tests/__init__.py",
        "tests/evals/__init__.py",
    ]

    for init_file in init_locations:
        path = Path(init_file)
        if not path.exists():
            path.write_text('"""Package initialization."""\n')
            logger.info("Created: %s", init_file)


def migrate_existing_files():
    """Move existing files to new locations."""
    migrations = [
        # (old_path, new_path)
        ("verifier.py", "src/verifier_primacy/core/verifier.py"),
        ("demo_mlx.py", "examples/demo_mlx.py"),
        ("cli_agent.py", "examples/cli_agent.py"),
    ]

    for old, new in migrations:
        old_path = Path(old)
        new_path = Path(new)

        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            logger.info("Copied: %s → %s", old, new)


def create_placeholder_docs():
    """Create placeholder documentation files."""
    docs = {
        "docs/architecture.md": "# Architecture\n\nTODO: Document architecture\n",
        "docs/api.md": "# API Reference\n\nTODO: Document API\n",
        "docs/backends.md": "# Backends\n\nTODO: Document backends\n",
    }

    for path, content in docs.items():
        p = Path(path)
        if not p.exists():
            p.write_text(content)
            logger.info("Created: %s", path)


def log_next_steps():
    """Log instructions for completing migration."""
    logger.info("\n" + "=" * 60)
    logger.info("Migration complete! Next steps:")
    logger.info("=" * 60)
    logger.info(
        """
1. Copy the Claude Code config files from the restructure package:
   - CLAUDE.md → ./CLAUDE.md
   - .claude/CLAUDE.md → ./.claude/CLAUDE.md
   - .claude/settings.json → ./.claude/settings.json
   - .claude/commands/*.md → ./.claude/commands/
   - .claude/agents/*.md → ./.claude/agents/
   - .claude/rules/*.md → ./.claude/rules/

2. Copy the new Python modules:
   - src/verifier_primacy/*.py files

3. Update pyproject.toml with new structure

4. Run tests to verify:
   uv sync
   uv run pytest

5. Commit the changes:
   git add -A
   git commit -m "Restructure for Claude Code optimization"

6. Test Claude Code integration:
   - Open project in Claude Code
   - Run /project:test
   - Run /project:demo
"""
    )


def main():
    logger.info("Verifier Primacy Migration Script")
    logger.info("-" * 40)

    create_directories()
    create_init_files()
    migrate_existing_files()
    create_placeholder_docs()
    log_next_steps()


if __name__ == "__main__":
    main()
