Run a quick demo of verifier_primacy capabilities:

1. Check if MLX is available (Apple Silicon)
2. Run the appropriate demo based on available backend
3. Show field-level confidence scores
4. Demonstrate verification catching invalid output

```bash
# Check backend availability
python -c "import mlx" 2>/dev/null && echo "MLX available" || echo "MLX not available"

# Run demo
uv run python examples/basic_extraction.py
```

Explain the output to help users understand:
- What confidence scores mean
- How verification rules constrain generation
- When human review is triggered
