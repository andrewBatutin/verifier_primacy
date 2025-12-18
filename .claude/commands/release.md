Prepare a release for verifier_primacy. Arguments: $ARGUMENTS (version number, e.g., "0.2.0")

1. **Pre-flight checks:**
   - All tests pass: `uv run pytest`
   - Type check passes: `uv run pyright`
   - No uncommitted changes: `git status --porcelain`

2. **Version bump:**
   - Update version in `pyproject.toml`
   - Update CHANGELOG.md with release notes

3. **Build and verify:**
   ```bash
   uv build
   uv run twine check dist/*
   ```

4. **Tag and commit:**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release v$ARGUMENTS"
   git tag -a "v$ARGUMENTS" -m "Release v$ARGUMENTS"
   ```

5. **Final summary:**
   - List all changes since last tag
   - Confirm ready for `git push && git push --tags`

Do NOT push automatically - wait for human confirmation.
