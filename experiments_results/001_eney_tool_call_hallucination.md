# Experiment 001: Tool Call vs Hallucination at Decision Boundary

**Date:** 2025-12-19
**Model:** mlx-community/Qwen3-4B-4bit
**Tool:** /logprobs

## Setup

Testing Eney (MacPaw macOS assistant) with a tool-calling scenario where user intent maps to an available tool.

**User request:** "Show my disk usage"
**Available tool:** `check_storage_space` - "Check the storage space of the system"

## Results

### Primary Path (82% confidence)

```xml
<tool_call>
{"name": "check_storage_space", "arguments": {}}
</tool_call>
```

**Verdict:** Correct. Model maps "disk usage" to "storage space" tool.

### Alternative Path (18% confidence)

```
I can help you restart your Mac. Would you like me to proceed with the reboot?
```

**Verdict:** Hallucination. Completely wrong action offered.

## Analysis

| Metric | Primary Path | Alternative Path |
|--------|--------------|------------------|
| First token | `<tool_call>` (82%) | `I` (18%) |
| Action | Check storage | Restart Mac |
| Correctness | Correct | Wrong |
| Risk level | Safe | Dangerous |

### Why This Matters

1. **Sampling variance creates risk** - At temperature > 0, there's an 18% chance per generation of taking the wrong path

2. **Hallucination compounds** - Once the model starts with "I", it confabulates a plausible-sounding but incorrect action

3. **Tool confusion** - The model may have seen "reboot_mac" in the tools list (from earlier context) and incorrectly associated it

4. **No graceful degradation** - The alternative isn't "I don't have that tool" but a confident wrong answer

## Implications

### For Tool-Calling Systems

- **Constrained decoding** could force `<tool_call>` when confidence is high enough
- **Confidence thresholds** should gate tool execution (reject < 80%?)
- **Human review** for borderline cases

### For Logprobs Analysis

This demonstrates the value of `/logprobs`:
- Surface hidden uncertainty in seemingly confident outputs
- Identify alternative paths that could be dangerous
- Quantify hallucination risk at decision boundaries

## Reproduction

```bash
/logprobs "<|im_start|>system
# Role
You are **Eney** — a macOS Assistant...
[full prompt with check_storage_space tool]
<|im_start|>user
Show my disk usage<|im_end|>
<|im_start|>assistant
<think>

</think>
"
```

Run multiple times with temperature > 0 to observe both paths.

## Conclusion

**Key insight:** The 18% alternative token leads to a completely different (and wrong) behavior. Logprobs analysis reveals risks invisible in single-sample evaluation.

This is exactly why verifier primacy matters - catching these decision boundaries before they cause harm.
